from collections.abc import Callable

from data_utils.utils import Nested_Dict, get_in, get_terminal_in


Filter = Callable[[Nested_Dict], bool]


def kibana_basic_filter(entry: Nested_Dict) -> bool:
    must_equals = [
        ("nodeRef.storeRef.protocol", "workspace"),
        ("type", "ccm:io"),
        ("properties.cm:edu_metadataset", "mds_oeh"),
    ]
    must_not_contain = [("aspects", "ccm:collection_io_reference")]

    must_holds = all(
        get_in(entry, key_seq.split("."), catch_errors=(KeyError, TypeError)) == value
        for key_seq, value in must_equals
    )

    lists = [
        get_in(entry, key_seq.split("."), catch_errors=(KeyError, TypeError))
        for key_seq, _ in must_not_contain
    ]

    if any(not isinstance(res_list, list) for res_list in lists):
        return False

    must_not_holds = all(
        value not in res_list
        for (
            _,
            value,
        ), res_list in zip(must_not_contain, lists)
    )

    return must_holds and must_not_holds


def kibana_publicly_visible(entry: Nested_Dict) -> bool:
    should_equals = [
        ("permissions.Read", "GROUP_EVERYONE"),
        ("collections.permissions.Read", "GROUP_EVERYONE"),
    ]

    return any(
        value in get_in(entry, key_seq.split(".")) for key_seq, value in should_equals
    )


def kibana_redaktionsbuffet(entry: Nested_Dict) -> bool:
    try:
        val: list[str] | None = get_in(
            entry,
            "collections.properties.cm:title".split("."),
            catch_errors=(KeyError,),
        )  # type: ignore
    except TypeError:
        print(
            get_in(
                entry,
                "collections.properties".split("."),
                catch_errors=tuple(),
            )
        )
        return False

    print(entry == val)

    if not val:
        return False

    return any(title == "Redaktionsbuffet" for title in val)
