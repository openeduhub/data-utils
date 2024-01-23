from collections.abc import Collection

dropped_values: dict[str, Collection[str]] = {
    "properties.ccm:taxonid": {
        "",
        "http://w3id.org/openeduhub/vocabs/discipline/Pädagogik",  # ambiguous
    },
    "properties.cclom:general_language": set(),
}

merged_values: dict[str, dict[str, str]] = {
    "properties.ccm:taxonid": {
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
    "properties.cclom:general_language_drop_region": {
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

skos_urls = {
    "properties.ccm:taxonid": "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json",
    "properties.ccm:educationalcontext": "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/educationalContext/index.json",
    "properties.ccm:fskRating": "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/fskRating/index.json",
}
