from baikpacking.agents.query_intent import _build_retrieval_intent_bundle, _classify_query_intent


def test_classify_query_intent_component_match():
    intent = _classify_query_intent("What tyres should I use for GranGuanche road?")

    assert intent.component == "tyres"
    assert "tyre" in intent.component_terms
    assert intent.asks_for_recommendation is True


def test_classify_query_intent_full_setup_fallback():
    intent = _classify_query_intent("Recommend a setup for North Cape Tarifa race")

    assert intent.component == "full_setup"
    assert intent.confidence == 0.25
    assert intent.component_terms == []


def test_classify_query_intent_broad_setup_with_bikepacking_keywords_stays_full_setup():
    intent = _classify_query_intent('Suggest a bikepacking setup for the "Granguanche Audax Gravel" event')

    assert intent.component == "full_setup"


def test_classify_query_intent_setup_request_stays_full_setup():
    intent = _classify_query_intent("Recommend a setup for Atlas Mountain Race")

    assert intent.component == "full_setup"


def test_build_retrieval_intent_bundle_for_component_query():
    intent = _classify_query_intent("What bags should I use?")
    descriptor = {
        "descriptor_query": "self-supported ultra endurance bikepacking race, Transiberica",
        "descriptor_query_with_intent": "self-supported ultra endurance bikepacking race, Transiberica. Question focus: What bags should I use?",
    }

    bundle = _build_retrieval_intent_bundle(descriptor=descriptor, intent=intent)

    assert bundle.broad_query == descriptor["descriptor_query"]
    assert bundle.include_component_query is True
    assert bundle.component_query is not None
    assert "bikepacking bags" in bundle.component_query


def test_build_retrieval_intent_bundle_for_full_setup():
    intent = _classify_query_intent("Recommend a complete setup")
    descriptor = {
        "descriptor_query": "self-supported ultra endurance bikepacking race, Tour Divide",
        "descriptor_query_with_intent": "self-supported ultra endurance bikepacking race, Tour Divide. Question focus: Recommend a complete setup",
    }

    bundle = _build_retrieval_intent_bundle(descriptor=descriptor, intent=intent)

    assert bundle.broad_query == descriptor["descriptor_query"]
    assert bundle.component_query == descriptor["descriptor_query_with_intent"]
    assert bundle.include_component_query is False


def test_classify_query_intent_bike_type_when_explicit_bike_choice():
    intent = _classify_query_intent("What bike should I use for Badlands?")

    assert intent.component == "bike_type"


def test_classify_query_intent_bike_type_for_explicit_comparison():
    intent = _classify_query_intent("Hardtail or gravel bike for Silk Road Mountain Race?")

    assert intent.component == "bike_type"
