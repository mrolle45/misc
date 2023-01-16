""" pointer.py

Defines pointer-like classes which when 'dereferenced' behave identically to
a target which it is pointing to.

The target can be any Python assignable:
- a name in the scope in which the pointer is created.
- an attribute of a given object.
- an subscript of a given object.
- possible enhancement: a subscript range of a given mutable sequence.
- possible enhancement: a tuple or list of targets, with a possible starred target.

"""

from __future__ import annotations

import collections.abc
from abc import *
import inspect
import operator

from typing import Generic, TypeVar, Callable
from typing_extensions import Self
from types import CellType

T = TypeVar('T')

class PointerType(ABC, Generic[T]):
	""" Base class defines the basic operations implemented by subclasses.
	The pointer can get, set, or delete any sort of target in the
	scope in which the pointer was created.

	A target is any expression which can appear on the left side of an assignment.
	This is one of:
	1. A name.  Note, special handling may be required for a local variable name.
		See the pointer_name subclass.
	2. An attribute of an object.  The object and the attribute name are evaluated
		(in that order) when creating the pointer.
	3. An item of a container object.  The container and the item key are evaluated
		(in that order) when creating the pointer.
	4. A tuple or list of targets.  One of these targets may be starred.
	5. A dereference of another pointer.

	A pointer is dereferenced in any of the following ways:
	1. pointer.target property.
		Get with 'pointer.target' expression
		Set with 'pointer.target = value' statement
		Delete with 'del pointer.target' statement

	2. Special methods.
		Get with pointer.__gettarget__()
		Set with pointer.__settarget__(value)
		Delete with pointer.__deltarget__()

		Any class which implements these three methods is a virtual subclass of PointerType.

	3. WITH COMPILER SUPPORT IN THE FUTURE, the expression (* pointer) will be
		same as (pointer.target), implemented by calling the appropriate method.
		This won't restrict the class of 'pointer' other than requiring the
		method be implemented.

	A pointer is created by calling a constructor to one of the pointer subclasses,
	the subclass depending on the nature of the target.

	WITH COMPILER SUPPORT IN THE FUTURE, any expression &(target) will create a
	pointer object of an implementation-dependent subclass of types.PointerType.
	For example, `*&ham.spam = 'eggs'` will have the same effect as `ham.spam = 'eggs'`.
	If `target` is a local name, there is no need for special handling, as is the
	case for the pointer_name subclass.

	Lifetime of pointer:  The pointer will have references to those objects used
	in its constructor.  Hence they will persist as long as the pointer does, even if
	the creator no longer exists.  If names used in the target are rebound or unbound
	in the creator, the pointer still uses the original objects.

	A pointer to a name in the creator scope persists after the end of that scope.  The
	dereferenced value can still be set or deleted.  This is the same behavior as when
	a function is defined which refers to that name; the new function refers to the name
	as a free variable.  In a class scope, this won't be possible because the function
	has no access to the names in the class; however, &name will work correctly.

	If you want to have, say, `ham` refer to the current binding of `ham` in the creator,
	then `ham` itself should be a pointer and the target should be (*ham).spam.  Note, if
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
		...

	@abstractmethod
	def __settarget__(self, value: T) -> None:
		...

	@abstractmethod
	def __deltarget__(self) -> None:
		...

	@classmethod
	def __subclasshook__(cls, C):
		if cls is PointerType:
			return collections.abc._check_methods(C, "__gettarget__", "__settarget__", "__deltarget__")
		return NotImplemented

class pointer_attr(PointerType[T]):
	""" Pointer to attribute {obj}.{name}. """

	def __init__(self, obj, name: str):
		self.obj, self.name = obj, name

	def __gettarget__(self) -> T: return getattr(self.obj, self.name)

	def __settarget__(self, value: T) -> None:
		setattr(self.obj, self.name, value)

	def __deltarget__(self) -> None:
		delattr(self.obj, self.name)

class pointer_item(PointerType[T]):
	""" Pointer to item {obj}[{key}]. """

	def __init__(self, obj, key):
		self.obj, self.key = obj, key

	def __gettarget__(self) -> T:
		return self.obj[self.key]

	def __settarget__(self, value: T) -> None:
		self.obj[self.key] = value

	def __deltarget__(self) -> None:
		del self.obj[self.key]

class pointer_cell(PointerType[T]):
	""" Pointer to a free variable in the creator's scope.
	This variable can be:
	1.	A local variable in an enclosing closed scope.
	2.	A local variable in the creator's scope,
		but ONLY if this is a closed scope.

	This is created by the constructor for pointer_name() when
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
		del self.cell.cell_contents


class pointer_name(PointerType[T]):
	""" Pointer to a free or global variable in the creator's scope,
	or a local variable in a creator's CLOSED scope.

	Constructor returns a pointer_cell for a free or local variable,
	and a pointer_item for a global variable.

	For a local variable in a creator's open scope, use pointer_local().
	"""
	def __new__(cls, func: Callable[[], T]) -> pointer_cell[T] | pointer_item[T]:
		""" Constructor with a function which returns the target in the caller.
		"""
		frame = inspect.currentframe().f_back
		stack = inspect.stack()
		code = stack[1].frame.f_code
		fcode = func.__code__
		dis.dis(func)
		if fcode.co_freevars:
			assert len(fcode.co_freevars) == 1, f'Too many free variables given to {cls.__name__}()'
			name = fcode.co_freevars[0]
			print(f'Making pointer to free name {name}')
			return pointer_cell(func.__closure__[0])
		else:
			assert len(fcode.co_names) == 1, f'Too many variable names given to {cls.__name__}()'
			name = fcode.co_names[0]
			print(f'Making pointer to global name {name}')
			return pointer_item(frame.f_globals, name)

	@classmethod
	def make_with_namespace(cls, name: Callable[[], T] | str) -> pointer_item[T]:
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

		print(f'Making pointer to {cls.name_type} name {name}')
		return pointer_item(cls.get_namespace(frame), name)

class pointer_local(pointer_name[T]):
	""" Pointer to local name in caller's scope.
	Must be an OPEN scope.
	"""
	def __new__(cls, name: Callable[[], T] | str) -> pointer_item[T]:
		return cls.make_with_namespace(name)

	get_namespace = operator.attrgetter('f_locals')
	name_type = 'local'
	
class pointer_global(pointer_name[T]):
	""" Pointer to global name in caller's scope.
	"""
	def __new__(cls, name: Callable[[], T] | str) -> pointer_item[T]:
		return cls.make_with_namespace(name)

	get_namespace = operator.attrgetter('f_globals')
	name_type = 'global'
	
class pointer_seq(PointerType[T]):
	""" A pointer to a sequence (tuple or list) of other pointers.
	One of them may be the singleton `pointer_star`.
	"""

# A singleton marker in a sequence of pointers used by pointer_seq.
# Indicates that the following pointer is starred.
class _star: pass
pointer_star = _star()

class C: pass
o = C()

p = pointer_attr(o, 'spam')

p.target = 'ham'
print(p.target)
del p.target

d = dict()
n = 'spam'
p = pointer_item(d, n)
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

def test(ptr):
	print(ptr.target)
	ptr.target = 42
	print(ptr.target)
	
# Code to be run by exec().
exsrc = '''
global glob
# No declaration for loc, which makes it "local".  The bytecode will be LOAD_NAME.


glob = 'g'
loc = 'l'

# The bytecodes in the lambdas are both LOAD_GLOBAL, because this is not the top level scope.
# loc is actually found in locals(), not globals().
test(pointer_name(lambda: glob))
test(pointer_local(lambda: loc))

glob = 'g again'
def f():
	print('in f()...')
	free = 'f'
	def g():
		loc = 'l'
		# Pointers to local, global, and free variables.
		# The bytecode in the global is LOAD_GLOBAL, because this it is (implicitly) global.
		# The bytecodes in the other lambdas are both LOAD_DEREF, because they are cell variables.
		for lam in ((lambda: glob), (lambda: loc), (lambda: free)):
			test(pointer_name(lam))
	g()
	free = 'f'
	outer = 'o'
	global glob
	glob = 'g yet again'

	class C:
		print('in C...')
		loc = 'l'
		# Pointers to local, global, and free variables.
		# The bytecode in the global is LOAD_GLOBAL, because this it is (implicitly) global.
		# The bytecodes in the other lambdas are both LOAD_DEREF, because they are cell variables.
		test(pointer_name(lambda: glob))
		test(pointer_local(lambda: loc))
		test(pointer_name(lambda: free))
		ptr = pointer_local(lambda: outer)

		outer = 'C.outer'
		test(ptr)
f()
'''

ex = compile(exsrc, '<exec src>', 'exec')
gdict = dict(globals(), x = 'glob x')
ldict = dict(locals(), x = 'loc x', __builtins__=dict(dummy=42))
import dis
import pdb
#pdb.run(exsrc, gdict, ldict)

exec(ex, gdict, ldict)


dis


