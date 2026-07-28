//! Cycle detection for recursive types.
//!
//! The visitors here ([`TypeTransformer`] and [`PairVisitor`]) are used in methods that
//! recursively visit types to transform them (e.g. [`Type::apply_type_mapping`]) or to
//! decide a relation between a pair of types (e.g. [`Type::has_relation_to`]).
//!
//! The typical pattern is that the "entry" method (e.g. [`Type::apply_type_mapping`]) will create
//! a visitor and pass it to the recursive method (e.g. [`Type::apply_type_mapping_impl`]).
//! Rust types that form part of a complex type (e.g. tuples, protocols, nominal instances, etc)
//! should usually just implement the recursive method, and all recursive calls should call the
//! recursive method and pass along the visitor.
//!
//! Not all recursive calls need to actually call `.visit` on the visitor; only when visiting types
//! that can create a recursive relationship (this includes, for example, type aliases and
//! protocols).
//!
//! There is a risk of double-visiting, for example if [`Type::apply_type_mapping_impl`] calls
//! `visitor.visit` when visiting a protocol type, and then internal `apply_type_mapping_impl`
//! methods of the Rust types implementing protocols also call `visitor.visit`. The best way to
//! avoid this is to prefer always calling `visitor.visit` only in the main recursive method on
//! `Type`.

use std::cell::{Cell, OnceCell, RefCell};
use std::cmp::Eq;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use ty_python_core::definition::Definition;

use crate::Db;
use crate::types::function::FunctionLiteral;
use crate::types::generics::Specialization;
use crate::types::visitor::{TypeCollector, TypeVisitor, walk_type_with_recursion_guard};
use crate::types::{
    BoundTypeVarInstance, ClassType, GenericAlias, NominalInstanceType, ProtocolInstanceType,
    SubclassOfType, Type, TypeAliasType, TypedDictType,
};

/// The type identity used for recursive checks/transformations.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum TypeIdentity<'db> {
    FunctionLiteral(FunctionLiteral<'db>),
    NewTypeInstance(Definition<'db>),
    NominalInstance(Definition<'db>),
    Protocol(Definition<'db>),
    TypeAlias(Definition<'db>),
    TypedDict(Definition<'db>),
    NonRecursive(Type<'db>),
}

impl<'db> Type<'db> {
    /// Projects a class-object wrapper onto the instance type whose definition owns its recursion.
    fn type_identity_owner(self, db: &'db dyn Db) -> Self {
        match self {
            Type::GenericAlias(alias) => Type::instance(db, ClassType::Generic(alias)),
            Type::SubclassOf(subclass_of) => subclass_of.to_instance(db),
            _ => self,
        }
    }

    pub(crate) fn to_type_identity(self, db: &'db dyn Db) -> TypeIdentity<'db> {
        let owner = self.type_identity_owner(db);
        if owner != self {
            return owner.to_type_identity(db);
        }

        match self {
            // Recursive TypeOf/CallableTypeOf expansion can attach different updated signatures
            // to the same function literal, so exact FunctionType equality is not stable.
            Type::FunctionLiteral(function) => TypeIdentity::FunctionLiteral(function.literal(db)),
            // A recursive NewType can have different eager base representations while it is
            // expanded, so exact type equality is not a stable recursion guard.
            Type::NewTypeInstance(newtype) => TypeIdentity::NewTypeInstance(newtype.definition(db)),
            Type::NominalInstance(instance) if instance.own_tuple_spec(db).is_none() => {
                instance.definition(db).map_or(
                    TypeIdentity::NonRecursive(self),
                    TypeIdentity::NominalInstance,
                )
            }
            Type::ProtocolInstance(protocol) => protocol
                .definition(db)
                .map_or(TypeIdentity::NonRecursive(self), TypeIdentity::Protocol),
            Type::TypeAlias(alias) => TypeIdentity::TypeAlias(alias.definition(db)),
            Type::TypedDict(typed_dict) => typed_dict
                .definition(db)
                .map_or(TypeIdentity::NonRecursive(self), TypeIdentity::TypedDict),
            _ => TypeIdentity::NonRecursive(self),
        }
    }

    /// Returns `false` if cheap inspection proves that `self` and `other` cannot have the same
    /// [`TypeIdentity`].
    ///
    /// This is a hot-path filter and must not walk either type. A `true` result is only a candidate
    /// match and must be confirmed with [`Type::to_type_identity`].
    pub(crate) fn may_share_type_identity(self, db: &'db dyn Db, other: Self) -> bool {
        if self == other {
            return true;
        }
        match (self, other) {
            (Type::FunctionLiteral(a), Type::FunctionLiteral(b)) => a.literal(db) == b.literal(db),
            (Type::NewTypeInstance(a), Type::NewTypeInstance(b)) => {
                a.definition(db) == b.definition(db)
            }
            (Type::NominalInstance(a), Type::NominalInstance(b)) => {
                a.definition(db) == b.definition(db)
            }
            (Type::ProtocolInstance(a), Type::ProtocolInstance(b)) => {
                a.definition(db) == b.definition(db)
            }
            (Type::TypeAlias(a), Type::TypeAlias(b)) => a.definition(db) == b.definition(db),
            (Type::TypedDict(a), Type::TypedDict(b)) => a.definition(db) == b.definition(db),
            (Type::GenericAlias(_) | Type::SubclassOf(_), _)
            | (_, Type::GenericAlias(_) | Type::SubclassOf(_)) => true,
            _ => false,
        }
    }

    /// Returns `false` if cheap inspection proves that `self` cannot grow from `active` in a type
    /// relation cycle.
    pub(crate) fn may_share_relation_type_identity(self, db: &'db dyn Db, active: Self) -> bool {
        if self == active {
            return true;
        }

        let specializations = match (self, active) {
            (Type::NominalInstance(current), Type::NominalInstance(active)) => {
                (current.specialization(db), active.specialization(db))
            }
            (Type::ProtocolInstance(current), Type::ProtocolInstance(active)) => {
                (current.specialization(db), active.specialization(db))
            }
            (Type::TypedDict(current), Type::TypedDict(active)) => {
                (current.specialization(db), active.specialization(db))
            }
            (Type::GenericAlias(_) | Type::SubclassOf(_), _)
            | (_, Type::GenericAlias(_) | Type::SubclassOf(_)) => return true,
            _ => return self.may_share_type_identity(db, active),
        };

        let (Some(current), Some(active)) = specializations else {
            return false;
        };
        // Inspect type arguments only after the identities match.
        current.generic_context(db) == active.generic_context(db)
    }

    /// Returns whether equal type identities form a recursive cycle.
    fn is_type_identity_cycle_with(self, db: &'db dyn Db, active: Self) -> bool {
        let self_owner = self.type_identity_owner(db);
        let active_owner = active.type_identity_owner(db);

        if self_owner != self || active_owner != active {
            return self_owner.is_type_identity_cycle_with(db, active_owner);
        }

        if self == active {
            return true;
        }
        match (self, active) {
            (Type::NominalInstance(..), Type::NominalInstance(..))
            | (Type::ProtocolInstance(..), Type::ProtocolInstance(..))
            | (Type::TypeAlias(..), Type::TypeAlias(..))
            | (Type::TypedDict(..), Type::TypedDict(..)) => !self.is_nested_in(db, active),
            _ => true,
        }
    }

    /// Returns whether equal type identities form a recursive cycle in a type relation.
    pub(crate) fn is_relation_type_identity_cycle_with(
        self,
        db: &'db dyn Db,
        active: Self,
    ) -> bool {
        let self_owner = self.type_identity_owner(db);
        let active_owner = active.type_identity_owner(db);

        if self_owner != self || active_owner != active {
            return self_owner.is_relation_type_identity_cycle_with(db, active_owner);
        }

        if self == active {
            return true;
        }
        match (self, active) {
            (Type::NominalInstance(current), Type::NominalInstance(active)) => {
                current.specialization_grows_from(db, active)
            }
            (Type::ProtocolInstance(current), Type::ProtocolInstance(active)) => {
                current.specialization_grows_from(db, active)
            }
            (Type::TypedDict(current), Type::TypedDict(active)) => {
                current.specialization_grows_from(db, active)
            }
            _ => self.is_type_identity_cycle_with(db, active),
        }
    }

    /// Returns whether `self` occurs within the specialization of `outer`.
    ///
    /// Descending from `Alias[Alias[int]]` to `Alias[int]` is finite even though both types have
    /// the same definition.
    fn is_nested_in(self, db: &'db dyn Db, outer: Self) -> bool {
        let specialization = match outer {
            Type::NominalInstance(instance) => instance.specialization(db),
            Type::TypeAlias(alias) => alias.specialization(db),
            Type::ProtocolInstance(protocol) => protocol.specialization(db),
            Type::TypedDict(typed_dict) => typed_dict.specialization(db),
            _ => None,
        };
        specialization.is_some_and(|specialization| {
            NestedTypeVisitor::contains_in_specialization(db, specialization, self)
        })
    }
}

struct NestedTypeVisitor<'db> {
    target: Type<'db>,
    active_aliases: ActiveRecursionDetector<Definition<'db>>,
    visited: TypeCollector<'db>,
    found: Cell<bool>,
}

impl<'db> NestedTypeVisitor<'db> {
    fn contains_in_specialization(
        db: &'db dyn Db,
        specialization: Specialization<'db>,
        target: Type<'db>,
    ) -> bool {
        let visitor = Self {
            target,
            active_aliases: ActiveRecursionDetector::default(),
            visited: TypeCollector::default(),
            found: Cell::new(false),
        };
        for ty in specialization.types(db) {
            visitor.visit_type(db, *ty);
            if visitor.found.get() {
                break;
            }
        }
        visitor.found.get()
    }
}

impl<'db> TypeVisitor<'db> for NestedTypeVisitor<'db> {
    fn should_visit_lazy_type_attributes(&self) -> bool {
        false
    }

    fn visit_type(&self, db: &'db dyn Db, ty: Type<'db>) {
        if self.found.get() {
            return;
        }

        if ty == self.target || ty.type_identity_owner(db) == self.target {
            self.found.set(true);
        } else {
            walk_type_with_recursion_guard(db, ty, self, &self.visited);
        }
    }

    fn visit_bound_type_var_type(
        &self,
        _db: &'db dyn Db,
        _bound_typevar: BoundTypeVarInstance<'db>,
    ) {
        // Bounds and defaults describe a type parameter; they are not contained in its argument.
    }

    fn visit_generic_alias_type(&self, db: &'db dyn Db, alias: GenericAlias<'db>) {
        self.visit_type(db, Type::instance(db, ClassType::Generic(alias)));
    }

    fn visit_subclass_of_type(&self, db: &'db dyn Db, subclass_of: SubclassOfType<'db>) {
        self.visit_type(db, subclass_of.to_instance(db));
    }

    fn visit_type_alias_type(&self, db: &'db dyn Db, alias: TypeAliasType<'db>) {
        if let Some(specialization) = alias.specialization(db) {
            for ty in specialization.types(db) {
                self.visit_type(db, *ty);
                if self.found.get() {
                    return;
                }
            }
        }

        let definition = alias.definition(db);
        self.active_aliases.visit(
            &definition,
            || {},
            || self.visit_type(db, alias.value_type(db)),
        );
    }

    fn visit_typed_dict_type(&self, db: &'db dyn Db, typed_dict: TypedDictType<'db>) {
        if let Some(specialization) = typed_dict.specialization(db) {
            for ty in specialization.types(db) {
                self.visit_type(db, *ty);
            }
        }
    }
}

struct EagerTypeContainmentVisitor<'db> {
    target: Type<'db>,
    visited: TypeCollector<'db>,
    found: Cell<bool>,
}

impl<'db> EagerTypeContainmentVisitor<'db> {
    fn contains(db: &'db dyn Db, ty: Type<'db>, target: Type<'db>) -> bool {
        let visitor = Self {
            target,
            visited: TypeCollector::default(),
            found: Cell::new(false),
        };
        visitor.visit_type(db, ty);
        visitor.found.get()
    }
}

impl<'db> TypeVisitor<'db> for EagerTypeContainmentVisitor<'db> {
    fn should_visit_lazy_type_attributes(&self) -> bool {
        false
    }

    fn visit_type(&self, db: &'db dyn Db, ty: Type<'db>) {
        if self.found.get() {
            return;
        }

        if ty == self.target {
            self.found.set(true);
        } else {
            walk_type_with_recursion_guard(db, ty, self, &self.visited);
        }
    }

    fn visit_bound_type_var_type(
        &self,
        _db: &'db dyn Db,
        _bound_typevar: BoundTypeVarInstance<'db>,
    ) {
        // Bounds and defaults describe a type parameter; they are not contained in its argument.
    }

    fn visit_type_alias_type(&self, db: &'db dyn Db, alias: TypeAliasType<'db>) {
        // The alias body is lazy, but its applied arguments are part of the containing type.
        if let Some(specialization) = alias.specialization(db) {
            for ty in specialization.types(db) {
                self.visit_type(db, *ty);
            }
        }
    }
}

impl<'db> Specialization<'db> {
    /// Returns whether this specialization wraps a corresponding active argument.
    ///
    /// This distinguishes an expanding recursive occurrence such as `Node[list[T]]` from a finite
    /// nested type such as `Node[Node[int]]`, whose inner specialization is smaller.
    fn grows_from(self, db: &'db dyn Db, active: Self) -> bool {
        if self.generic_context(db) != active.generic_context(db) {
            return false;
        }

        self.types(db)
            .iter()
            .zip(active.types(db))
            .any(|(current_ty, active_ty)| {
                current_ty != active_ty
                    && EagerTypeContainmentVisitor::contains(db, *current_ty, *active_ty)
            })
    }
}

impl<'db> NominalInstanceType<'db> {
    fn definition(self, db: &'db dyn Db) -> Option<Definition<'db>> {
        self.class(db).definition(db)
    }

    fn specialization(self, db: &'db dyn Db) -> Option<Specialization<'db>> {
        if self.own_tuple_spec(db).is_some() {
            return None;
        }
        self.class(db)
            .into_generic_alias()
            .map(|alias| alias.specialization(db))
    }

    fn specialization_grows_from(self, db: &'db dyn Db, active: Self) -> bool {
        self.specialization(db)
            .zip(active.specialization(db))
            .is_some_and(|(current, active)| current.grows_from(db, active))
    }
}

impl<'db> ProtocolInstanceType<'db> {
    fn definition(self, db: &'db dyn Db) -> Option<Definition<'db>> {
        let (origin, _) = self.class_origin()?.static_class_literal(db)?;
        Some(origin.definition(db))
    }

    fn specialization(self, db: &'db dyn Db) -> Option<Specialization<'db>> {
        self.class_origin()
            .and_then(|class| class.into_generic_alias())
            .map(|alias| alias.specialization(db))
    }

    fn specialization_grows_from(self, db: &'db dyn Db, active: Self) -> bool {
        self.specialization(db)
            .zip(active.specialization(db))
            .is_some_and(|(current, active)| current.grows_from(db, active))
    }
}

impl<'db> TypedDictType<'db> {
    fn specialization(self, db: &'db dyn Db) -> Option<Specialization<'db>> {
        self.defining_class()
            .and_then(ClassType::into_generic_alias)
            .map(|alias| alias.specialization(db))
    }

    fn specialization_grows_from(self, db: &'db dyn Db, active: Self) -> bool {
        self.specialization(db)
            .zip(active.specialization(db))
            .is_some_and(|(current, active)| current.grows_from(db, active))
    }
}

/// An item that provides the identity used to detect active recursive cycles.
pub trait HasIdentity<'db> {
    type Id: PartialEq;

    /// Returns `false` if cheap inspection proves that `self` and `other` cannot have the same
    /// identity.
    ///
    /// This method must not walk either item. Returning `true` does not imply that the identities
    /// match; [`HasIdentity::to_identity`] confirms it.
    fn may_share_identity(&self, _db: &'db dyn Db, _other: &Self) -> bool {
        true
    }

    /// Returns an identity that remains stable while this item is active in a [`CycleDetector`].
    fn to_identity(&self, db: &'db dyn Db) -> Self::Id;

    /// Returns whether two equal identities form a recursive cycle.
    fn is_identity_cycle_with(&self, _db: &'db dyn Db, _active: &Self) -> bool {
        true
    }
}

impl<'db> HasIdentity<'db> for Type<'db> {
    type Id = TypeIdentity<'db>;

    fn may_share_identity(&self, db: &'db dyn Db, other: &Self) -> bool {
        self.may_share_type_identity(db, *other)
    }

    fn to_identity(&self, db: &'db dyn Db) -> Self::Id {
        Type::to_type_identity(*self, db)
    }

    fn is_identity_cycle_with(&self, db: &'db dyn Db, active: &Self) -> bool {
        self.is_type_identity_cycle_with(db, *active)
    }
}

pub(crate) type PairVisitor<'db, Tag, C> = CycleDetector<'db, Tag, (Type<'db>, Type<'db>), C, 1>;

impl<'db> HasIdentity<'db> for (Type<'db>, Type<'db>) {
    type Id = (TypeIdentity<'db>, TypeIdentity<'db>);

    fn may_share_identity(&self, db: &'db dyn Db, other: &Self) -> bool {
        self.0.may_share_relation_type_identity(db, other.0)
            && self.1.may_share_relation_type_identity(db, other.1)
    }

    fn to_identity(&self, db: &'db dyn Db) -> Self::Id {
        (self.0.to_type_identity(db), self.1.to_type_identity(db))
    }

    fn is_identity_cycle_with(&self, db: &'db dyn Db, active: &Self) -> bool {
        self.0.is_relation_type_identity_cycle_with(db, active.0)
            && self.1.is_relation_type_identity_cycle_with(db, active.1)
    }
}

impl<'db, Context> HasIdentity<'db> for (Type<'db>, Context, Type<'db>)
where
    Context: Copy + PartialEq,
{
    type Id = (TypeIdentity<'db>, Context, TypeIdentity<'db>);

    fn may_share_identity(&self, db: &'db dyn Db, other: &Self) -> bool {
        self.0.may_share_relation_type_identity(db, other.0)
            && self.1 == other.1
            && self.2.may_share_relation_type_identity(db, other.2)
    }

    fn to_identity(&self, db: &'db dyn Db) -> Self::Id {
        (
            self.0.to_type_identity(db),
            self.1,
            self.2.to_type_identity(db),
        )
    }

    fn is_identity_cycle_with(&self, db: &'db dyn Db, active: &Self) -> bool {
        self.0.is_relation_type_identity_cycle_with(db, active.0)
            && self.2.is_relation_type_identity_cycle_with(db, active.2)
    }
}

/// `CycleDetector` is temporary, so callers should choose the capacity that keeps observed cycle
/// paths inline even when that makes `seen` slightly larger than an `FxIndexSet<T>`.
#[derive(Debug)]
pub struct CycleDetector<'db, Tag, T: HasIdentity<'db>, R, const INLINE_CAPACITY: usize> {
    /// The active recursion stack and the lazily-computed identity of each item.
    /// Completed visits are removed from the end of the stack.
    seen: RefCell<SmallVec<[ActiveCycleDetectorVisit<'db, T>; INLINE_CAPACITY]>>,

    /// Memoized results from earlier visits in the current recursive operation.
    cache: RefCell<CycleDetectorCache<T, R>>,

    fallback: R,

    _tag: PhantomData<fn() -> &'db Tag>,
}

impl<'db, Tag, T, R, const INLINE_CAPACITY: usize> CycleDetector<'db, Tag, T, R, INLINE_CAPACITY>
where
    T: HasIdentity<'db>,
{
    pub fn new(fallback: R) -> Self {
        CycleDetector {
            seen: RefCell::new(SmallVec::new()),
            cache: RefCell::new(CycleDetectorCache::new()),
            fallback,
            _tag: PhantomData,
        }
    }
}

impl<'db, Tag, T, R: Clone, const INLINE_CAPACITY: usize>
    CycleDetector<'db, Tag, T, R, INLINE_CAPACITY>
where
    T: Hash + Eq + Clone + HasIdentity<'db>,
{
    #[inline]
    pub fn visit(&self, db: &'db dyn Db, item: T, compute: impl FnOnce() -> R) -> R {
        match self.begin_visit(db, item) {
            CycleDetectorVisit::Ready(result) => result,
            CycleDetectorVisit::Cycle(_) => self.fallback.clone(),
            CycleDetectorVisit::Pending(item) => {
                let result = compute();
                self.finish_visit(item, result)
            }
        }
    }

    /// Visits `item`, returning it in `Err` if another active item has the same identity.
    ///
    /// The caller must convert `Err(item)` into an operation-specific conservative result. An
    /// exact recursive reentry uses the detector's configured fallback and is returned as `Ok`.
    #[inline]
    pub(super) fn try_visit(
        &self,
        db: &'db dyn Db,
        item: T,
        compute: impl FnOnce() -> R,
    ) -> Result<R, T> {
        match self.begin_visit(db, item) {
            CycleDetectorVisit::Ready(result) => Ok(result),
            CycleDetectorVisit::Cycle(item) => Err(item),
            CycleDetectorVisit::Pending(item) => {
                let result = compute();
                Ok(self.finish_visit(item, result))
            }
        }
    }

    fn begin_visit(&self, db: &'db dyn Db, item: T) -> CycleDetectorVisit<T, R> {
        if let Some(result) = self.cache.borrow().get(&item) {
            return CycleDetectorVisit::Ready(result.clone());
        }

        let seen = self.seen.borrow();
        if seen.iter().any(|active| active.item == item) {
            return CycleDetectorVisit::Ready(self.fallback.clone());
        }

        let mut candidates = seen
            .iter()
            .filter(|active| item.may_share_identity(db, &active.item))
            .peekable();
        let identity = if candidates.peek().is_none() {
            OnceCell::new()
        } else {
            // Defer constructing identities until a candidate match shows that another active
            // item could form a cycle.
            let identity = item.to_identity(db);
            if candidates.any(|active| {
                active.identity.get_or_init(|| active.item.to_identity(db)) == &identity
                    && item.is_identity_cycle_with(db, &active.item)
            }) {
                return CycleDetectorVisit::Cycle(item);
            }
            OnceCell::from(identity)
        };
        drop(seen);

        self.seen.borrow_mut().push(ActiveCycleDetectorVisit {
            item: item.clone(),
            identity,
        });
        CycleDetectorVisit::Pending(item)
    }

    /// Finish a [`CycleDetectorVisit::Pending`] visit and cache its result.
    fn finish_visit(&self, item: T, result: R) -> R {
        let active = self.seen.borrow_mut().pop();
        debug_assert!(active.as_ref().is_some_and(|active| active.item == item));
        self.cache
            .borrow_mut()
            .insert_completed(item, result.clone());
        result
    }
}

struct ActiveCycleDetectorVisit<'db, T: HasIdentity<'db>> {
    item: T,
    identity: OnceCell<T::Id>,
}

impl<'db, T: fmt::Debug + HasIdentity<'db>> fmt::Debug for ActiveCycleDetectorVisit<'db, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.item.fmt(f)
    }
}

/// Result of starting a cycle-detector visit.
pub(super) enum CycleDetectorVisit<T, R> {
    /// The item already has a completed result or hit an exact recursive edge.
    Ready(R),
    /// A different item with the same abstract identity is already pending.
    Cycle(T),
    /// The caller should compute the result and finish the pending visit.
    Pending(T),
}

/// Guards recursive type transformations.
pub(crate) struct TypeTransformer<'db, Tag> {
    /// The active transformation stack and its recursive identities.
    /// Completed visits are removed from the end of the stack.
    seen: RefCell<SmallVec<[ActiveTypeTransformation<'db>; 3]>>,

    /// Memoized transformations from earlier visits in the current recursive operation.
    cache: RefCell<CycleDetectorCache<Type<'db>, Type<'db>>>,

    _tag: PhantomData<fn() -> Tag>,
}

impl<Tag> Default for TypeTransformer<'_, Tag> {
    fn default() -> Self {
        Self {
            seen: RefCell::default(),
            cache: RefCell::default(),
            _tag: PhantomData,
        }
    }
}

impl<'db, Tag> TypeTransformer<'db, Tag> {
    #[inline]
    pub(crate) fn visit_type(
        &self,
        db: &'db dyn Db,
        ty: Type<'db>,
        compute: impl FnOnce() -> Type<'db>,
    ) -> Type<'db> {
        match self.begin_visit(db, ty) {
            TypeTransformerVisit::Ready(result) => result,
            TypeTransformerVisit::Pending(ty) => {
                let result = compute();
                self.finish_visit(ty, result)
            }
        }
    }

    fn begin_visit(&self, db: &'db dyn Db, ty: Type<'db>) -> TypeTransformerVisit<'db> {
        if let Some(result) = self.cache.borrow().get(&ty) {
            return TypeTransformerVisit::Ready(*result);
        }

        let seen = self.seen.borrow();
        if seen.iter().any(|active| active.ty == ty) {
            return TypeTransformerVisit::Ready(ty);
        }

        let mut candidates = seen
            .iter()
            .filter(|active| ty.may_share_type_identity(db, active.ty))
            .peekable();
        let identity = if candidates.peek().is_none() {
            OnceCell::new()
        } else {
            let identity = ty.to_type_identity(db);
            if candidates.any(|active: &ActiveTypeTransformation<'db>| {
                active
                    .identity
                    .get_or_init(|| active.ty.to_type_identity(db))
                    == &identity
                    && ty.is_type_identity_cycle_with(db, active.ty)
            }) {
                return TypeTransformerVisit::Ready(ty);
            }
            OnceCell::from(identity)
        };
        drop(seen);

        self.seen
            .borrow_mut()
            .push(ActiveTypeTransformation { ty, identity });
        TypeTransformerVisit::Pending(ty)
    }

    fn finish_visit(&self, ty: Type<'db>, result: Type<'db>) -> Type<'db> {
        let active = self.seen.borrow_mut().pop();
        debug_assert_eq!(active.map(|active| active.ty), Some(ty));
        self.cache.borrow_mut().insert_completed(ty, result);
        result
    }
}

#[derive(Debug)]
struct ActiveTypeTransformation<'db> {
    ty: Type<'db>,
    identity: OnceCell<TypeIdentity<'db>>,
}

enum TypeTransformerVisit<'db> {
    Ready(Type<'db>),
    Pending(Type<'db>),
}

impl<'db, Tag, T, R: Default, const INLINE_CAPACITY: usize> Default
    for CycleDetector<'db, Tag, T, R, INLINE_CAPACITY>
where
    T: HasIdentity<'db>,
{
    fn default() -> Self {
        CycleDetector::new(R::default())
    }
}

/// The memoized results for a [`CycleDetector`].
///
/// Most populated cycle-detector caches contain at most two results. Keep those results inline,
/// but spill on the third distinct result so lookups in wider caches remain hashed.
#[derive(Debug, Default)]
enum CycleDetectorCache<T, R> {
    #[default]
    Empty,
    One((T, R)),
    Two([(T, R); 2]),
    Spilled(FxHashMap<T, R>),
}

impl<T, R> CycleDetectorCache<T, R> {
    const fn new() -> Self {
        Self::Empty
    }

    fn get(&self, item: &T) -> Option<&R>
    where
        T: Hash + Eq,
    {
        match self {
            Self::Empty => None,
            Self::One((cached_item, result)) => (cached_item == item).then_some(result),
            Self::Two(entries) => entries
                .iter()
                .find_map(|(cached_item, result)| (cached_item == item).then_some(result)),
            Self::Spilled(cache) => cache.get(item),
        }
    }

    /// Inserts a completed item after the caller has checked that `item` is not already cached.
    fn insert_completed(&mut self, item: T, result: R)
    where
        T: Hash + Eq,
    {
        debug_assert!(self.get(&item).is_none());
        self.insert_new(item, result);
    }

    fn insert_new(&mut self, item: T, result: R)
    where
        T: Hash + Eq,
    {
        let entry = (item, result);
        *self = match mem::replace(self, Self::Empty) {
            Self::Empty => Self::One(entry),
            Self::One(first) => Self::Two([first, entry]),
            Self::Two(entries) => Self::spill(entries, entry),
            Self::Spilled(mut cache) => {
                cache.insert(entry.0, entry.1);
                Self::Spilled(cache)
            }
        };
    }

    #[cold]
    fn spill(entries: [(T, R); 2], third: (T, R)) -> Self
    where
        T: Hash + Eq,
    {
        Self::Spilled(entries.into_iter().chain([third]).collect())
    }

    #[cfg(test)]
    const fn is_spilled(&self) -> bool {
        matches!(self, Self::Spilled(_))
    }
}

/// Recursion detection without memoization.
///
/// This is useful when a recursive relation needs a coinductive-style "we're already proving this
/// goal, assume it for now" step, but completed results are not safe to reuse for future visits to
/// the same abstract key.
#[derive(Debug)]
pub(crate) struct ActiveRecursionDetector<T> {
    seen: RefCell<FxHashSet<T>>,
}

impl<T> Default for ActiveRecursionDetector<T> {
    fn default() -> Self {
        Self {
            seen: RefCell::new(FxHashSet::default()),
        }
    }
}

impl<T: Hash + Eq + Clone> ActiveRecursionDetector<T> {
    pub(crate) fn visit<R>(
        &self,
        item: &T,
        on_cycle: impl FnOnce() -> R,
        func: impl FnOnce() -> R,
    ) -> R {
        if !self.seen.borrow_mut().insert(item.clone()) {
            return on_cycle();
        }

        // Keep the active-recursion state scoped even if `func` unwinds. In some cases, we catch
        // panics and continue handling later work on the same thread.
        let _guard = ActiveRecursionGuard {
            seen: &self.seen,
            item,
        };

        func()
    }
}

struct ActiveRecursionGuard<'a, T: Hash + Eq> {
    seen: &'a RefCell<FxHashSet<T>>,
    item: &'a T,
}

impl<T: Hash + Eq> Drop for ActiveRecursionGuard<'_, T> {
    fn drop(&mut self) {
        self.seen.borrow_mut().remove(self.item);
    }
}

#[cfg(test)]
mod tests {
    use super::{CycleDetector, CycleDetectorVisit, Db, HasIdentity};
    use crate::db::tests::setup_db;
    use std::cell::Cell;
    use std::hash::{Hash, Hasher};

    struct TestVisit;

    type Detector<'db> = CycleDetector<'db, TestVisit, u8, u8, 1>;

    impl<'db> HasIdentity<'db> for u8 {
        type Id = Self;

        fn to_identity(&self, _db: &'db dyn Db) -> Self::Id {
            *self
        }
    }

    #[derive(Clone)]
    struct CountingIdentityItem<'a> {
        value: u8,
        identity_calls: &'a Cell<usize>,
    }

    impl<'a> CountingIdentityItem<'a> {
        const fn new(value: u8, identity_calls: &'a Cell<usize>) -> Self {
            Self {
                value,
                identity_calls,
            }
        }
    }

    impl PartialEq for CountingIdentityItem<'_> {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Eq for CountingIdentityItem<'_> {}

    impl Hash for CountingIdentityItem<'_> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.value.hash(state);
        }
    }

    impl<'db> HasIdentity<'db> for CountingIdentityItem<'_> {
        type Id = u8;

        fn may_share_identity(&self, _db: &'db dyn Db, other: &Self) -> bool {
            self.value % 2 == other.value % 2
        }

        fn to_identity(&self, _db: &'db dyn Db) -> Self::Id {
            self.identity_calls.set(self.identity_calls.get() + 1);
            self.value
        }
    }

    #[derive(Clone, Eq, Hash, PartialEq)]
    struct ConstantIdentityItem(u8);

    impl<'db> HasIdentity<'db> for ConstantIdentityItem {
        type Id = ();

        fn to_identity(&self, _db: &'db dyn Db) -> Self::Id {}
    }

    #[test]
    fn caches_results_and_spills_after_two_entries() {
        let db = setup_db();
        let detector = Detector::new(0);

        assert_eq!(detector.visit(&db, 1, || 10), 10);
        assert_eq!(detector.visit(&db, 1, || 40), 10);
        assert_eq!(detector.visit(&db, 2, || 20), 20);
        assert!(!detector.cache.borrow().is_spilled());
        assert_eq!(detector.visit(&db, 3, || 30), 30);
        assert!(detector.cache.borrow().is_spilled());

        assert_eq!(detector.visit(&db, 2, || 40), 20);
        assert_eq!(detector.visit(&db, 3, || 40), 30);
    }

    #[test]
    fn nested_visit_short_circuits_on_cycle() {
        let db = setup_db();
        let detector = Detector::new(0);

        assert_eq!(
            detector.visit(&db, 1, || detector.visit(&db, 1, || 20) + 10),
            10
        );
    }

    #[test]
    fn computes_each_active_identity_once() {
        let db = setup_db();
        let identity_calls = Cell::new(0);
        let detector = CycleDetector::<TestVisit, CountingIdentityItem<'_>, u8, 1>::new(0);

        assert_eq!(
            detector.visit(&db, CountingIdentityItem::new(1, &identity_calls), || {
                detector.visit(&db, CountingIdentityItem::new(3, &identity_calls), || 1)
            }),
            1
        );
        assert_eq!(identity_calls.get(), 2);
    }

    #[test]
    fn skips_identity_for_distinct_candidates() {
        let db = setup_db();
        let identity_calls = Cell::new(0);
        let detector = CycleDetector::<TestVisit, CountingIdentityItem<'_>, u8, 1>::new(0);

        assert_eq!(
            detector.visit(&db, CountingIdentityItem::new(1, &identity_calls), || {
                detector.visit(&db, CountingIdentityItem::new(2, &identity_calls), || 1)
            }),
            1
        );
        assert_eq!(identity_calls.get(), 0);
    }

    #[test]
    fn skips_identity_without_a_distinct_active_item() {
        let db = setup_db();
        let identity_calls = Cell::new(0);
        let detector = CycleDetector::<TestVisit, CountingIdentityItem<'_>, u8, 1>::new(0);

        assert_eq!(
            detector.visit(&db, CountingIdentityItem::new(1, &identity_calls), || 1),
            1
        );
        assert_eq!(
            detector.visit(&db, CountingIdentityItem::new(1, &identity_calls), || 2),
            1
        );
        assert_eq!(identity_calls.get(), 0);
    }

    #[test]
    fn different_items_with_same_identity_form_cycle() {
        let db = setup_db();
        let detector = CycleDetector::<TestVisit, ConstantIdentityItem, u8, 1>::new(0);

        let CycleDetectorVisit::Pending(pending) =
            detector.begin_visit(&db, ConstantIdentityItem(1))
        else {
            panic!("the first identity should be pending");
        };
        let CycleDetectorVisit::Cycle(item) = detector.begin_visit(&db, ConstantIdentityItem(2))
        else {
            panic!("a different item with the same identity should form a cycle");
        };
        assert_eq!(item.0, 2);
        detector.finish_visit(pending, 1);

        let CycleDetectorVisit::Ready(seen) = detector.begin_visit(&db, ConstantIdentityItem(1))
        else {
            panic!("the first identity should be ready after the pending visit is finished");
        };
        assert_eq!(seen, 1);
        let CycleDetectorVisit::Pending(pending) =
            detector.begin_visit(&db, ConstantIdentityItem(2))
        else {
            panic!("the second identity should be pending after the first is finished");
        };
        detector.finish_visit(pending, 2);
        let CycleDetectorVisit::Ready(seen) = detector.begin_visit(&db, ConstantIdentityItem(2))
        else {
            panic!("the second identity should be ready after the pending visit is finished");
        };
        assert_eq!(seen, 2);
    }
}
