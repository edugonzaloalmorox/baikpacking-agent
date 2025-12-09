from baikpacking.agents.recommender_agent import recommend_setup

if __name__ == "__main__":
    query = "I'm doing Gran Guanche, prefer comfort over aero. Give me recommendations."
    rec = recommend_setup(query)

    print("SUMMARY:")
    print(rec.summary)

    print("\nBIKE TYPE:", rec.bike_type)
    print("WHEELS:", rec.wheels)
    print("TYRES:", rec.tyres)
    print("DRIVETRAIN:", rec.drivetrain)

    print("\nBAGS:", rec.bags)
    print("SLEEP SYSTEM:", rec.sleep_system)

    print("\nSIMILAR RIDERS:")
    for r in rec.similar_riders:
        print(f"- {r.name} @ {r.event_title} (year={r.year}, score={r.best_score:.4f})")
