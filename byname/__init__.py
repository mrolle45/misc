""" byname package.

Defines byname-like classes which refer to a target expression in the scope in which
they were created.  Dereferencing them is equivalent to evaluating the target.

The target can be assigned or deleted if this operation is valid in the creator's frame.
The target can be:
- a name in the creator's frame.
- an attribute of a given object.
- an subscript or slice of a given object.
- a tuple or list of targets, with a possible starred target.
	Delete not valid with a starred target.
- dereference of a byname object.

"""

from __future__ import annotations

import collections.abc
from abc import *
import inspect
from operator import attrgetter
from itertools import chain
import dis
from dataclasses import dataclass, field
import ast

from typing import Generic, TypeVar, Callable, Iterable
from typing_extensions import Self
from types import CellType

T = TypeVar('T')

class BynameType(ABC, Generic[T]):
	""" Base class defines the basic operations implemented by subclasses.
	The byname can get, set, or delete any sort of target in the
	scope in which the byname was created.

	A target is any expression which can appear on the left side of an assignment.
	This is one of:
	1. A name.  Note, special handling may be required for a local variable name.
		See the byname_name subclass.
	2. An attribute of an object.  The object and the attribute name are evaluated
		(in that order) when creating the byname.
	3. An item of a container object.  The container and the item key are evaluated
		(in that order) when creating the byname.
	4. A tuple or list of targets.  One of these targets may be starred.
	5. A dereference of another byname.

	A byname is dereferenced in any of the following ways:
	1. byname.target property.
		Get with 'byname.target' expression
		Set with 'byname.target = value' statement
		Delete with 'del byname.target' statement

	2. Special methods.
		Get with byname.__gettarget__()
		Set with byname.__settarget__(value)
		Delete with byname.__deltarget__()

		Any class which implements these three methods is a virtual subclass of BynameType.

	3. WITH COMPILER SUPPORT IN THE FUTURE, the expression (* byname) will be
		same as (byname.target), implemented by calling the appropriate method.
		This won't restrict the class of 'byname' other than requiring the
		method be implemented.

	A byname is created by calling a constructor to one of the byname subclasses,
	the subclass depending on the nature of the target.

	WITH COMPILER SUPPORT IN THE FUTURE, any expression &(target) will create a
	byname object of an implementation-dependent subclass of types.BynameType.
	For example, `*&ham.spam = 'eggs'` will have the same effect as `ham.spam = 'eggs'`.
	If `target` is a local name, there is no need for special handling, as is the
	case for the byname_name subclass.

	Lifetime of byname:  The byname will have references to those objects used
	in its constructor.  Hence they will persist as long as the byname does, even if
	the creator no longer exists.  If names used in the target are rebound or unbound
	in the creator, the byname still uses the original objects.

	A byname to a name in the creator frame persists after the end of that frame.  The
	dereferenced value can still be set or deleted.  This is the same behavior as when
	a function is defined which refers to that name; the new function refers to the name
	as a free variable.  In a class scope, this won't be possible because the function
	has no access to the names in the class; however, &name will work correctly.

	If you want to have, say, `ham` refer to the current binding of `ham` in the creator,
	then `ham` itself should be a byname and the target should be (*ham).spam.  Note, if
	you save `(*ham)` in some variable, as in `x = *ham`, then `x` will NOT change if
	`*ham` later changes.

	"""

	@property
	def target(self) -> T:
		return self.__gettarget__()

	@target.setter
	def target(self, value: T) -> None:
		self.__settarget__(value)

	@target.deleter
	def target(self) -> None:
		self.__deltarget__()

	@abstractmethod
	def __gettarget__(self) -> T:
		raise TypeError

	@abstractmethod
	def __settarget__(self, value: T) -> None:
		raise TypeError

	@abstractmethod
	def __deltarget__(self) -> None:
		raise TypeError

	@classmethod
	def __subclasshook__(cls, C):
		if cls is BynameType:
			return collections.abc._check_methods(C, "__gettarget__", "__settarget__", "__deltarget__")
		return NotImplemented

	starred: ClassVar[bool] = False
	exc: ClassVar[Type] = Exception

class byname_attr(BynameType[T]):
	""" Byname to attribute {obj}.{name}. """

	def __init__(self, obj, name: str):
		self.obj, self.name = obj, name

	def __gettarget__(self) -> T: return getattr(self.obj, self.name)

	def __settarget__(self, value: T) -> None:
		setattr(self.obj, self.name, value)

	def __deltarget__(self) -> None:
		delattr(self.obj, self.name)

	exc: ClassVar[Type] = AttributeError

class byname_item(BynameType[T]):
	""" Byname to item {obj}[{key}]. """

	def __init__(self, obj, key):
		self.obj, self.key = obj, key

	def __gettarget__(self) -> T:
		return self.obj[self.key]

	def __settarget__(self, value: T) -> None:
		self.obj[self.key] = value

	def __deltarget__(self) -> None:
		del self.obj[self.key]

	exc: ClassVar[Type] = KeyError

class byname_slice(byname_item[T]):
	""" Byname to item {obj}[{slice}]. """

	def __init__(self, obj, *args):
		""" Slice of given sequence object. """
		if len(args) == 1:
			args = (0, args[0])
		if len(args) == 2:
			args = (*args, 1)
		assert len(args) == 3, 'Wrong number of arguments to byname_slice().'
		super().__init__(obj, *args)

class byname_cell(BynameType[T]):
	""" Byname to a free variable in the creator's scope.
	This variable can be:
	1.	A local variable in an enclosing closed scope.
	2.	A local variable in the creator's scope,
		but ONLY if this is a closed scope.

	This is created by the constructor for byname_name() when
	the name is one of the above.

	It can also be created directly from one of the elements of
	func.__closure__ which represents the free variable in a
	function.
	"""

	def __init__(self, cell: CellType):
		self.cell = cell

	def __gettarget__(self) -> T:
		return self.cell.cell_contents

	def __settarget__(self, value: T) -> None:
		self.cell.cell_contents = value

	def __deltarget__(self) -> None:
		self.cell.cell_contents				# Raise exception if already deleted.
		del self.cell.cell_contents

	exc: ClassVar[Type] = ValueError

class byname_name(BynameType[T]):
	""" Byname to a free or global variable in the creator's scope,
	or a local variable in a creator's CLOSED scope.

	Constructor returns a byname_cell for a free or local variable,
	and a byname_item for a global variable.

	For a local variable in a creator's open scope, use byname_local().
	"""
	def __new__(cls, func: Callable[[], T]) -> byname_cell[T] | byname_item[T]:
		""" Constructor with a function which returns the target in the caller.
		"""
		fr = inspect.currentframe().f_back
		stack = inspect.stack()
		code = stack[1].frame.f_code
		fcode = func.__code__
		dis.dis(func)
		if fcode.co_freevars:
			assert len(fcode.co_freevars) == 1, f'Too many free variables given to {cls.__name__}()'
			name = fcode.co_freevars[0]
			locs = fr.f_locals
			globs = fr.f_globals
			b = fr.f_back
			print(f'Making byname to free name {name}')
			return byname_cell(func.__closure__[0])
		else:
			assert len(fcode.co_names) == 1, f'Too many variable names given to {cls.__name__}()'
			name = fcode.co_names[0]
			print(f'Making byname to global name {name}')
			return byname_item(fr.f_globals, name)

	@classmethod
	def make_with_namespace(cls, name: Callable[[], T] | str) -> byname_item[T]:
		# Have to go up two frames because we are called by the cls() constructor.
		frame = inspect.currentframe().f_back.f_back
		if callable(name):
			stack = inspect.stack()
			fcode = name.__code__
			dis.dis(name)
			code = stack[1].frame.f_code
			if fcode.co_freevars:
				assert len(fcode.co_freevars) == 1, f'Too many free variables given to {cls.__name__}()'
				name = fcode.co_freevars[0]
			else:
				assert len(fcode.co_names) == 1, f'Too many variable names given to {cls.__name__}()'
				name = fcode.co_names[0]
		else:
			assert isinstance(name, str), f'{cls.__name__} requires a str, not {type(str).__name}.'

		print(f'Making byname to {cls.name_type} name {name}')
		return byname_item(cls.get_namespace(frame), name)

	exc: ClassVar[Type] = NameError

class byname_local(byname_name[T]):
	""" Byname to local name in caller's scope.
	Must be an OPEN scope.
	"""
	def __new__(cls, name: Callable[[], T] | str) -> byname_item[T]:
		return cls.make_with_namespace(name)

	get_namespace = attrgetter('f_locals')
	name_type = 'local'
	
class byname_global(byname_name[T]):
	""" Byname to global name in caller's scope.
	"""
	def __new__(cls, name: Callable[[], T] | str) -> byname_item[T]:
		return cls.make_with_namespace(name)

	get_namespace = attrgetter('f_globals')
	name_type = 'global'
	
class byname_seq(BynameType[Iterable[T]]):
	""" A byname to a sequence (tuple or list) of other items, which are bynames.

	Some of them may be byname_star objects.
	A byname with stars cannot be deleted.
	A byname with starts can be assigned is there is only one of them.

	Getting the target returns a tuple, or list, of the bynames' targets.
	Setting the target will accept any iterable of values that has a len().
	Deleting the target will delete each byname's target.  Not supported if there is a *ptr.

	"""
	setter: Callable[Iterable[T], None]

	def __init__(self, *bynames: BynameType[T]):
		next_start: int = 0
		groups: list[list[BynameType[T]]] = []
		stars: list[BynameType[T]] = []
		for pos, ptr in enumerate(bynames):
			if ptr.starred:
				groups.append(bynames[next_start : pos])
				stars.append(ptr)
				next_start = pos + 1
		groups.append(bynames[next_start : ])

		if stars:
			if len(stars) > 1:
				# Too many stars for setter.
				def setter(*args):
					raise TypeError('Cannot set byname with multiple starred items')
			else:
				def setter(value: Iterable[T], min_len = len(bynames) - 1,
						   heads = list(map(attrgetter('__settarget__'), groups[0])),
						   head_len = len(groups[0]),
						   star_setter = stars[0].ptr.__settarget__,
						   tails = list(map(attrgetter('__settarget__'), groups[1])),
						   tail_len = len(groups[1]),
						   ) -> None:
					vals = list(value)
					assert len(vals) >= min_len, f'Expected at least {min_len} values to unpack, received {len(vals)}.'
					any(setter(val) for setter, val in zip(heads, vals[ : head_len]))
					star_setter(vals[head_len : - tail_len])
					any(setter(val) for setter, val in zip(tails, vals[- tail_len:]))
			def deleter():
				raise TypeError('Cannot delete byname with any starred items')
		else:
			def setter(value: Iterable[T], exact_len = len(bynames) - 1,
						setters = list(map(attrgetter('__settarget__'), bynames)),
						) -> None:
				vals = list(value)
				assert len(vals) == exact_len, f'Expected exactly {exact_len} values to unpack, received {len(vals)}.'
				any(setter(val) for setter, val in zip(setters, vals))
			def deleter(
						deleters = list(map(attrgetter('__deltarget__'), bynames)),
						) -> None: 
				any(deleter() for deleter in deleters)

		def groupgetter(group):
			""" returns Function(no args) -> iterable of targets of group members. """
			getters = tuple(map(attrgetter('__gettarget__'), group))
			def getter(getters = getters) -> Iterable[T]:
				yield from (getter() for getter in getters)
			return getter

		def itergetters():
			yield groupgetter(groups[0])
			for group, star in zip(groups[1:], stars):
				yield lambda: (star.ptr.__gettarget__())
				yield groupgetter(group)

		def getter(getters = list(itergetters()), seq = self.seq_type) -> Iterable[T]:
			return seq(chain(*(getter() for getter in getters)))

		self.getter = getter
		self.setter = setter
		self.deleter = deleter

		self.ptrs = list(bynames)
		self.len = len(bynames)

	def __gettarget__(self) -> T:
		return self.getter()

	def __settarget__(self, value: Iterable[T]) -> None:
		self.setter(value)

	def __deltarget__(self) -> None:
		self.deleter()

	seq_type: ClassVar[Type] = tuple

	def make_tuple(self): return byname_seq(*self.ptrs)
	def make_list(self): return byname_list(*self.ptrs)

class byname_list(byname_seq[T]):

	seq_type: ClassVar[Type] = list

class byname_star(BynameType[T]):
	""" Wraps another byname and indicates that it is a starred target.
	"""
	def __init__(self, byname: BynameType[T]):
		self.ptr = byname
	
	def __gettarget__(self) -> T:
		return NotImplemented

	def __settarget__(self, value: T) -> None:
		...

	def __deltarget__(self) -> None:
		...

	starred: ClassVar[bool] = True

def rewrite(script: str) -> str:
	""" Replaces &target, *byname, and &param with byname calls.
	In a def or lambda, references to params are replaced with byname calls.
	"""
	import io, ast
	from tokenize import generate_tokens, untokenize, TokenInfo, NAME, OP, STAR, AMPER

	# 1.  Clean up the input so that it is syntactically valid.
	#	Replace & and * with + (but not in a def or lambda).
	#	Will parse as either binary or unary op.  We'll catch the unary ops later.
	#	Remove & in def and lambda parameters.  Make note of the parameter names.

	@dataclass
	class Func:
		""" Describes a def statement or lambda expression.
		"""
		srow: int					# starting line of the keyword
		scol: int					# starting position of the keyword
		params: dict[str, int] = field(default_factory=dict)	# Parameter names introduced with '&'.
									# [name] = how many times.

		def add(self, name: str, count: int = 1) -> None:
			if name not in self.params: self.params[name] = 0
			self.params[name] += count

		def __bool__(self) -> bool: return bool(self.params)

	# Functions with any &param in them.
	funcs: list[Func]
	funcs = []

	# Any & or * tokens.  Don't know if they are part of a binary op or not.
	@dataclass
	class Op:
		""" Describes a & or * token.
		"""
		srow: int					# starting line of the keyword
		scol: int					# starting position of the keyword
		type: int					# exact type of token (STAR or AMPER)

	ops: list[Op]
	ops = []

	def gen1(tokens: Iterable[tokenize.TokenInfo]) -> Iterable[tokenize.TokenInfo]:
		""" Make some replacements in token stream and make lists of altered tokens and functions. """
		it: Iterator[tokenize.TokenInfo] = iter(tokens)
		nonlocal funcs, ops
		try:
			while True:
				tok = next(it)
				if tok.type is NAME:
					if tok.string in ('def', 'lambda'):
						# This is a function def or lambda.
						func = Func(*tok.start)
						count = 0					# How many '&' before next parameter
						while tok.string != ':':
							if tok.string == '&':
								count += 1
								tok = None			# Don't emit this token
							elif tok.type is NAME:
								# This is parameter name if any '&' preceded it.
								if count:
									func.add(tok.string, count)
									count = 0

							if tok: yield tok
							tok = next(it)

						if func:
							funcs.append(func)
				elif tok.type is OP:
					if tok.string in '&*':
						tok = tok._replace(string='+')
						ops.append(Op(*tok.start, tok.exact_type))
				yield tok

		except StopIteration: pass

	tokens = list(gen1(generate_tokens(io.StringIO(script).readline)))

	# 2. Transform the tokens into an AST tree.
	#	In a function or lambda body, replace params with derefences.
	#	Anywhere, including the above, replace * and & operators with
	#	byname constructors, based on operator and context.
	#	Binary * and & were replaced with + and will need to be put back to the original. 

	scr1 = untokenize(tokens)
	tree = ast.parse(scr1)

	return tokens

class RewriteAST(ast.NodeTransformer):

	def __init__(funcs, ops):
		self.funcs = funcs
		self.ops = ops


scr = '''
def f(a, &b, &&c): pass
x = &[a, b, c]
y = *x
'''

s = rewrite(scr)

p = byname_list(
	byname_global(lambda: a),
	byname_star(byname_global(lambda: b)),
	byname_global(lambda: c))
p.target = range(4)
print(a, b, c)				# a = 0, b = [1, 2], c = 3
print(p.target)
pd = byname_list(
	byname_global(lambda: a),
	byname_global(lambda: b),
	byname_global(lambda: c))
del pd.target
print(list(globals()))

class C: pass
o = C()

p = byname_attr(o, 'spam')

p.target = 'ham'
print(p.target)
del p.target

d = dict()
n = 'spam'
p = byname_item(d, n)
p.target = 'ham'
print(p.target)
del p.target

gl = 0

def f():
	fr = 1
	def g():
		lo = 2
		return lambda: (gl, lo, fr)
	class C:
		lo = 2
		lam = lambda: (gl, __class__.lo, fr)

	return g(), C

lam, C = f()

lam

def foo():
	x = 1

#def check_ptr(ptr):
#	print(ptr.target)
#	ptr.target = 42
#	print(ptr.target)
	
# Code to be run by exec().
#exsrc = '''
#global glob
## No declaration for loc, which makes it "local".  The bytecode will be LOAD_NAME.


#glob = 'g'
#loc = 'l'

## The bytecodes in the lambdas are both LOAD_GLOBAL, because this is not the top level scope.
## loc is actually found in locals(), not globals().
#check_ptr(byname_name(lambda: glob))
#check_ptr(byname_local(lambda: loc))

#glob = 'g again'
#def f():
#	print('in f()...')
#	free = 'f'
#	def g():
#		loc = 'l'
#		# Bynames to local, global, and free variables.
#		# The bytecode in the global is LOAD_GLOBAL, because this it is (implicitly) global.
#		# The bytecodes in the other lambdas are both LOAD_DEREF, because they are cell variables.
#		for lam in ((lambda: glob), (lambda: loc), (lambda: free)):
#			check_ptr(byname_name(lam))
#	g()
#	free = 'f'
#	outer = 'o'
#	global glob
#	glob = 'g yet again'

#	class C:
#		print('in C...')
#		loc = 'l'
#		# Bynames to local, global, and free variables.
#		# The bytecode in the global is LOAD_GLOBAL, because this it is (implicitly) global.
#		# The bytecodes in the other lambdas are both LOAD_DEREF, because they are cell variables.
#		check_ptr(byname_name(lambda: glob))
#		check_ptr(byname_local(lambda: loc))
#		check_ptr(byname_name(lambda: free))
#		ptr = byname_local(lambda: outer)

#		outer = 'C.outer'
#		check_ptr(ptr)
#f()
#'''

#ex = compile(exsrc, '<exec src>', 'exec')
#gdict = dict(globals(), x = 'glob x')
#ldict = dict(locals(), x = 'loc x', __builtins__=dict(dummy=42))
#import dis
#import pdb
##pdb.run(exsrc, gdict, ldict)

#exec(ex, gdict, ldict)


#dis


