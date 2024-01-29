from collections.abc import Collection
from enum import Enum

from data_utils.utils import Basic_Value, Basic_Value_Not_None


class Fields(Enum):
    ID = "nodeRef.id"
    TITLE = "properties.cclom:title"
    DESCRIPTION = "properties.cclom:general_description"
    LANGUAGE = "properties.cclom:general_language"
    TAXONID = "properties.ccm:taxonid"
    EDUCATIONAL_CONTEXT = "properties.ccm:educationalcontext"
    FSK_RATING = "properties.ccm:fskRating"
    COLLECTIONS = "collections.properties.cm:title"


dropped_values: dict[str, Collection[Basic_Value_Not_None]] = {
    Fields.TAXONID.value: {
        "",
        "http://w3id.org/openeduhub/vocabs/discipline/???",
        "http://w3id.org/openeduhub/vocabs/discipline/Pädagogik",  # ambiguous
    },
    Fields.LANGUAGE.value: set(),
}

remapped_values: dict[str, dict[Basic_Value_Not_None, Basic_Value]] = {
    Fields.TAXONID.value: {
        "http://w3id.org/openeduhub/vocabs/discipline/Darstellendes-Spiel": "http://w3id.org/openeduhub/vocabs/discipline/12002",
        "http://w3id.org/openeduhub/vocabs/discipline/Deutsch": "http://w3id.org/openeduhub/vocabs/discipline/120",
        "http://w3id.org/openeduhub/vocabs/discipline/Deutsch als Zweitsprache": "http://w3id.org/openeduhub/vocabs/discipline/28002",
        "http://w3id.org/openeduhub/vocabs/discipline/Deutsch als": "http://w3id.org/openeduhub/vocabs/discipline/28002",
        "Zweitsprache": "http://w3id.org/openeduhub/vocabs/discipline/28002",
        "http://w3id.org/openeduhub/vocabs/discipline/Englisch": "http://w3id.org/openeduhub/vocabs/discipline/20001",
        "http://w3id.org/openeduhub/vocabs/discipline/Geografie": "http://w3id.org/openeduhub/vocabs/discipline/220",
        "http://w3id.org/openeduhub/vocabs/discipline/Geschichte": "http://w3id.org/openeduhub/vocabs/discipline/240",
        "http://w3id.org/openeduhub/vocabs/discipline/Informatik": "http://w3id.org/openeduhub/vocabs/discipline/320",
        "http://w3id.org/openeduhub/vocabs/discipline/Mathematik": "http://w3id.org/openeduhub/vocabs/discipline/380",
        "http://w3id.org/openeduhub/vocabs/discipline/Physik": "http://w3id.org/openeduhub/vocabs/discipline/460",
        "http://w3id.org/openeduhub/vocabs/discipline/Religion": "http://w3id.org/openeduhub/vocabs/discipline/520",
        "http://w3id.org/openeduhub/vocabs/discipline/Spanisch": "http://w3id.org/openeduhub/vocabs/discipline/20007",
        "http://w3id.org/openeduhub/vocabs/discipline/Medienbildung": "http://w3id.org/openeduhub/vocabs/discipline/900",
        "http://w3id.org/openeduhub/vocabs/discipline/Physik": "http://w3id.org/openeduhub/vocabs/discipline/460",
        "http://w3id.org/openeduhub/vocabs/discipline/Mathematik": "http://w3id.org/openeduhub/vocabs/discipline/380",
    },
    Fields.LANGUAGE.value: {
        "de_DE": "de",
        "de_AT": "de",
        "DE": "de",
        "de-DE": "de",
        "Deutsch": "de",
        "en-US-LEARN": "en",
        "en_US": "en",
        "en_GB": "en",
        "hu_HU": "hu",
        "es_CR": "es",
        "es_ES": "es",
        "es_AR": "es",
        "fr_FR": "fr",
        "tr_TR": "tr",
        "latin": "la",
    },
}

skos_urls: dict[str, str] = {
    Fields.TAXONID.value: "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json",
    Fields.EDUCATIONAL_CONTEXT.value: "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/educationalContext/index.json",
    Fields.FSK_RATING.value: "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/fskRating/index.json",
}
