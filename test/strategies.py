from collections.abc import Callable, Collection, Sequence
from functools import partial
from typing import Any, Optional, TypeVar

import hypothesis.strategies as st
from data_utils.data import Basic_Value, Data_Point, Terminal_Value

# values that may be associated with any key / added to a list
basic_values_not_none = st.one_of(
    st.text(),
    st.integers(),
    st.floats(),
)
basic_values = st.one_of(
    st.text(),
    st.integers(),
    st.floats(),
    st.none(),
)
nested_basic_values = st.one_of(basic_values, st.lists(basic_values))

T = TypeVar("T")


@st.composite
def filtered_strategy(
    draw: st.DrawFn,
    strategy: st.SearchStrategy[T],
    blacklist: Collection[T] = frozenset(),
    whitelist: Collection[T] = frozenset(),
) -> T:
    if len(whitelist) > 0:
        strategy = st.sampled_from(list(whitelist))

    return draw(
        strategy.filter(
            lambda x: x not in blacklist
            if not isinstance(x, list)
            else all(x_sub not in blacklist for x_sub in x)
        )
    )


filtered_nested_basic_values = partial(filtered_strategy, strategy=nested_basic_values)


def cut_subtree(base_dict: Data_Point, key: str) -> Data_Point:
    """Remove a key-value pair from a data-point."""
    return Data_Point(
        {
            base_key: base_value
            for base_key, base_value in base_dict.items()
            if base_key != key
        }
    )


@st.composite
def replace_subtree(
    draw: st.DrawFn,
    base_dict: Data_Point,
    key: str,
    blacklist: Collection[Basic_Value] = frozenset(),
    whitelist: Collection[Basic_Value] = frozenset(),
) -> Data_Point:
    """Replace the value for the given key with a random new value."""
    # make sure that the new value does not equal the old one
    new_value = draw(
        filtered_nested_basic_values(blacklist=blacklist, whitelist=whitelist)
    )
    return Data_Point(
        {
            base_key: base_value if base_key != key else new_value
            for base_key, base_value in base_dict.items()
        }  # type: ignore
    )


@st.composite
def add_garbage(
    draw: st.DrawFn,
    base_dict: Data_Point,
    value_blacklist: Collection[Basic_Value] = frozenset(),
    value_whitelist: Collection[Basic_Value] = frozenset(),
    key_blacklist: Collection[str] = frozenset(),
    key_whitelist: Collection[str] = frozenset(),
) -> Data_Point:
    """Add a new key-value pair."""
    key = draw(
        filtered_strategy(
            st.text(),
            blacklist=set(base_dict.keys()) | set(key_blacklist),
            whitelist=key_whitelist,
        )
    )
    value = draw(
        filtered_nested_basic_values(
            blacklist=value_blacklist, whitelist=value_whitelist
        )
    )

    return Data_Point(
        {base_key: base_value for base_key, base_value in base_dict.items()}
        | {key: value}  # type: ignore
    )


@st.composite
def recursively_mutated_dict(
    draw: st.DrawFn,
    base_dict: Data_Point,
    key: str,
    index: int | None = None,
    **kwargs,
) -> Data_Point:
    """
    Randomly mutate the sub-tree starting at the given key.

    If index is given, the value of the key is assumed to be
    a list of sub-trees instead. Here, we randomly mutate the sub-tree
    at the position within the list given by index.
    """
    followed_dict = base_dict[key]
    if index is None:
        assert isinstance(followed_dict, dict)
        return Data_Point(
            {
                base_key: base_value
                if base_key != key
                else draw(mutated_data_point(followed_dict, **kwargs))
                for base_key, base_value in base_dict.items()
            }
        )

    assert isinstance(followed_dict, list)
    followed_dict = followed_dict[index]
    assert isinstance(followed_dict, dict)
    return Data_Point(
        {
            base_key: base_value
            if base_key != key
            else draw(mutated_data_point(followed_dict, **kwargs))
            for base_key, base_value in base_dict.items()
        }
    )


@st.composite
def mutated_data_point(
    draw: st.DrawFn,
    base_dict: Data_Point,
    allow_cut=True,
    allow_replace=True,
    allow_add_key=True,
    allow_add_to_list=True,
    allow_remove_from_list=True,
    value_whitelist: Collection[Any] = frozenset(),
    value_blacklist: Collection[Any] = frozenset(),
    mod_key_whitelist: Collection[str] = frozenset(),
    mod_key_blacklist: Collection[str] = frozenset(),
    add_key_whitelist: Collection[str] = frozenset(),
    add_key_blacklist: Collection[str] = frozenset(),
    min_iters: int = 1,
    max_iters: int = 1,
) -> Data_Point:
    """Randomly change some aspect(s) of the given data-point."""
    kwargs = {
        "allow_cut": allow_cut,
        "allow_replace": allow_replace,
        "allow_add_key": allow_add_key,
        "allow_add_to_list": allow_add_to_list,
        "allow_remove_from_list": allow_remove_from_list,
        "value_whitelist": value_whitelist,
        "value_blacklist": value_blacklist,
    }
    iters = draw(st.integers(min_iters, max_iters))
    for _ in range(iters):
        actions: list[Callable[[], Data_Point]] = list()
        # add a random new key with a random terminal value
        if allow_add_key:
            actions.append(
                lambda: draw(
                    add_garbage(
                        base_dict,
                        value_blacklist=value_blacklist,
                        value_whitelist=value_whitelist,
                        key_blacklist=add_key_blacklist,
                        key_whitelist=add_key_whitelist,
                    )
                )
            )

        if base_dict:
            key = draw(
                st.sampled_from(
                    list(base_dict.keys())
                    if len(mod_key_whitelist) == 0
                    else list(mod_key_whitelist)
                ).filter(lambda x: x not in mod_key_blacklist)
            )
            value = base_dict[key]

            # remove this key and its value
            if allow_cut:
                actions.append(lambda: cut_subtree(base_dict, key))
            # keep the key but replace its value with a random terminal value
            if allow_replace:
                actions.append(
                    lambda: draw(
                        replace_subtree(
                            base_dict,
                            key,
                            blacklist=value_blacklist,
                            whitelist=value_whitelist,
                        )
                    )
                )
            # recursively mutate the underlying nested dictionary
            if isinstance(value, dict):
                actions.append(
                    lambda: draw(recursively_mutated_dict(base_dict, key, **kwargs))
                )

            elif isinstance(value, list):
                # add a random value to the list
                if allow_add_to_list:
                    added_value = draw(
                        filtered_strategy(
                            basic_values,
                            blacklist=value_blacklist,
                            whitelist=value_whitelist,
                        )
                    )
                    actions.append(
                        lambda: Data_Point(
                            {
                                base_key: base_value
                                if base_key != key
                                else value + [added_value]  # type: ignore
                                for base_key, base_value in base_dict.items()
                            }
                        )
                    )

                if len(value) > 0:
                    index = draw(st.sampled_from(range(len(value))))

                    # remove a random value from the list
                    if allow_remove_from_list:
                        value_without_index = [
                            x for i, x in enumerate(value) if i != index
                        ]
                        actions.append(
                            lambda: Data_Point(
                                {
                                    base_key: base_value
                                    if base_key != key
                                    else value_without_index
                                    for base_key, base_value in base_dict.items()
                                }
                            )
                        )

                    # recursively mutate a random subtree from the list
                    if isinstance(value[index], dict):
                        actions.append(
                            lambda: draw(
                                recursively_mutated_dict(
                                    base_dict, key, index, **kwargs
                                )
                            )
                        )

        action = draw(st.sampled_from(actions))
        base_dict = action()

    return base_dict
