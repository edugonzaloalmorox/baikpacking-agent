from types import SimpleNamespace

from baikpacking.agents.models import SetupCore, SetupRecommendation
from baikpacking.agents.output_validation import _fill_requested_component_from_riders
from baikpacking.agents.orchestration_models import RecommendationPolicy


def test_fill_requested_component_from_structured_rider_field():
    rec = SetupRecommendation(summary="x", reasoning="y")
    riders = [SimpleNamespace(tyres="45mm slick/semi-slick", chunks=[])]

    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="tyres")

    assert out.recommended_setup.tyres == "45mm slick/semi-slick"


def test_fill_requested_component_from_chunk_when_structured_missing():
    rec = SetupRecommendation(summary="x", reasoning="y")
    riders = [
        SimpleNamespace(
            wheels=None,
            chunks=[SimpleNamespace(text="Wheelset: 700c alloy rims with dynamo hub")],
        )
    ]

    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="wheels")

    assert out.recommended_setup.wheels == "700c alloy rims"
    assert "Grounded examples for wheels" in (out.recommended_setup.notes or "")
    assert "Wheelset: 700c alloy rims with dynamo hub" in (out.recommended_setup.notes or "")


def test_tyres_backfill_rejects_repair_kit_text():
    rec = SetupRecommendation(summary="x", reasoning="y")
    riders = [
        SimpleNamespace(
            tyres=None,
            chunks=[SimpleNamespace(text="Tubeless repair kit with sealant and pump")],
        )
    ]

    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="tyres")

    assert out.recommended_setup.tyres is None
    assert out.recommended_setup.notes is None


def test_fill_full_setup_populates_supported_fields_and_examples():
    rec = SetupRecommendation(summary="Generic summary", reasoning="Generic reasoning")
    riders = [
        SimpleNamespace(
            bike_type="Gravel bike",
            wheels="700c alloy rims with dynamo hub",
            tyres="45mm semi-slick tyres with tubeless setup",
            drivetrain="GRX 1x with power meter",
            bags="Apidura frame bag with top tube bag",
            sleep_system="bivy with groundsheet",
            chunks=[
                SimpleNamespace(text="Apidura frame bag with top tube bag"),
                SimpleNamespace(text="GRX 1x drivetrain with power meter"),
            ],
        ),
        SimpleNamespace(
            bike_type="Hardtail",
            wheels="29er",
            tyres="40mm tubeless",
            drivetrain="SRAM Rival",
            bags="Tailfin seat pack",
            sleep_system="quilt",
            chunks=[
                SimpleNamespace(text="Tailfin seat pack"),
                SimpleNamespace(text="quilt sleep system"),
            ],
        ),
    ]

    policy = RecommendationPolicy(
        mode="strict_grounded",
        allow_specific_brands=True,
        allow_specific_specs=True,
        allow_event_specific_claims=True,
    )
    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="full_setup", policy=policy)

    assert out.recommended_setup.bike_type == "Gravel bike"
    assert out.recommended_setup.wheels == "700c alloy rims"
    assert out.recommended_setup.tyres == "45mm semi-slick tyres"
    assert out.recommended_setup.drivetrain == "GRX 1x"
    assert out.recommended_setup.bags == "Apidura frame bag"
    assert out.recommended_setup.sleep_system == "bivy"
    assert "Grounded examples" in (out.recommended_setup.notes or "")
    assert "Apidura frame bag" in (out.recommended_setup.notes or "")
    assert "Grounded rider patterns support the filled setup fields." in out.summary
    assert "similar event" not in (out.summary or "").lower()
    assert "similar event" not in (out.reasoning or "").lower()


def test_fill_component_query_adds_multiple_examples_and_pattern_summary():
    rec = SetupRecommendation(summary="Generic summary", reasoning="Generic reasoning")
    riders = [
        SimpleNamespace(tyres="45mm semi-slick", chunks=[SimpleNamespace(text="45mm semi-slick setup")]),
        SimpleNamespace(tyres="40mm tubeless", chunks=[SimpleNamespace(text="40mm tubeless setup")]),
        SimpleNamespace(tyres="42mm", chunks=[SimpleNamespace(text="42mm fast-rolling tyres")]),
    ]

    policy = RecommendationPolicy(
        mode="pattern_based",
        allow_specific_brands=False,
        allow_specific_specs=True,
        allow_event_specific_claims=False,
    )
    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="tyres", policy=policy)

    assert out.recommended_setup.tyres == "45mm semi-slick"
    assert "Grounded examples for tyres" in (out.recommended_setup.notes or "")
    assert "45mm semi-slick" in (out.recommended_setup.notes or "")
    assert "40mm tubeless" in (out.recommended_setup.notes or "")
    assert any(word in out.reasoning.lower() for word in ["cluster", "mixed"])


def test_fill_component_query_does_not_introduce_unsupported_specificity():
    rec = SetupRecommendation(summary="Generic summary", reasoning="Generic reasoning")
    riders = [
        SimpleNamespace(
            bags=None,
            chunks=[SimpleNamespace(text="navigation unit, water bottles, and frame bag")],
        )
    ]

    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="bags")

    assert out.recommended_setup.bags is None
    assert "Grounded examples" not in (out.recommended_setup.notes or "")


def test_bike_only_request_prunes_other_fields():
    rec = SetupRecommendation(
        summary="x",
        reasoning="y",
        recommended_setup=SetupCore(
            bike_type=None,
            wheels="700c alloy rims",
            tyres="45mm semi-slick",
            drivetrain="GRX 1x",
            bags="frame bag",
            sleep_system="bivy",
            lighting="front light",
            navigation="gps unit",
            water_capacity="2L",
        ),
    )
    riders = [SimpleNamespace(bike_type="Gravel bike", chunks=[SimpleNamespace(text="Gravel bike")])]
    policy = RecommendationPolicy(
        mode="strict_grounded",
        allow_specific_brands=True,
        allow_specific_specs=True,
        allow_event_specific_claims=True,
    )

    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="bike_type", policy=policy)

    assert out.recommended_setup.bike_type == "Gravel bike"
    assert out.recommended_setup.wheels is None
    assert out.recommended_setup.tyres is None
    assert out.recommended_setup.drivetrain is None
    assert out.recommended_setup.bags is None
    assert out.recommended_setup.sleep_system is None
    assert out.recommended_setup.lighting is None
    assert out.recommended_setup.navigation is None
    assert out.recommended_setup.water_capacity is None
    assert "Grounded examples for bike_type" in (out.recommended_setup.notes or "")


def test_generic_fallback_keeps_outputs_sparse_and_drops_spillover():
    rec = SetupRecommendation(summary="x", reasoning="y")
    riders = [
        SimpleNamespace(
            bags=None,
            drivetrain=None,
            chunks=[SimpleNamespace(text="navigation unit, water bottles, frame bag, and GRX drivetrain")],
        )
    ]
    policy = RecommendationPolicy(
        mode="generic_fallback",
        allow_specific_brands=False,
        allow_specific_specs=False,
        allow_event_specific_claims=False,
    )

    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="bags", policy=policy)

    assert out.recommended_setup.bags is None
    assert out.recommended_setup.notes is None


def test_wheels_backfill_rejects_full_bike_sentence():
    rec = SetupRecommendation(summary="Generic summary", reasoning="Generic reasoning")
    riders = [
        SimpleNamespace(
            wheels=None,
            chunks=[
                SimpleNamespace(
                    text="Full bike setup with GRX drivetrain, 45mm tyres, frame bag, water storage and navigation."
                )
            ],
        )
    ]

    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="wheels")

    assert out.recommended_setup.wheels is None


def test_wheels_backfill_rejects_mixed_bike_drivetrain_text():
    rec = SetupRecommendation(summary="x", reasoning="y")
    riders = [
        SimpleNamespace(
            wheels=None,
            chunks=[SimpleNamespace(text="700c wheelset with GRX drivetrain and 45mm tyres")],
        )
    ]

    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="wheels")

    assert out.recommended_setup.wheels is None


def test_sleep_system_backfill_rejects_navigation_and_bag_text():
    rec = SetupRecommendation(summary="Generic summary", reasoning="Generic reasoning")
    riders = [
        SimpleNamespace(
            sleep_system=None,
            chunks=[
                SimpleNamespace(
                    text="Navigation unit, frame bag, water bladder, and lights for the bike."
                )
            ],
        )
    ]

    out = _fill_requested_component_from_riders(rec=rec, riders=riders, query_component="sleep_system")

    assert out.recommended_setup.sleep_system is None
    assert out.recommended_setup.notes is None
