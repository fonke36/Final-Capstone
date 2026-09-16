import random
from collections import Counter


# ---------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------

SEED = 42
random.seed(SEED)


# ---------------------------------------------------------
# 2. Required categories and statuses
# ---------------------------------------------------------

CATEGORIES = [
    "Billing",
    "Technical Issue",
    "Account Access",
    "Product Defect",
    "General Inquiry",
]

STATUSES = [
    "Open",
    "In Progress",
    "Escalated",
    "Resolved",
    "Closed",
]


# ---------------------------------------------------------
# 3. Dataset design choices
# ---------------------------------------------------------

TOTAL_TICKETS = 50

# Category weights used for the additional random tickets.
CATEGORY_WEIGHTS = {
    "Billing": 0.25,
    "Technical Issue": 0.25,
    "Account Access": 0.20,
    "Product Defect": 0.15,
    "General Inquiry": 0.15,
}

# Status weights used for randomly generated statuses.
STATUS_WEIGHTS = {
    "Open": 0.20,
    "In Progress": 0.25,
    "Escalated": 0.10,
    "Resolved": 0.30,
    "Closed": 0.15,
}

# Escalation rate is deliberately fixed at 20%.
ESCALATION_RATE = 0.20


# ---------------------------------------------------------
# 4. Realistic resolution-time ranges
# ---------------------------------------------------------

RESOLUTION_RANGES = {
    "Billing": (4, 24),
    "Technical Issue": (6, 48),
    "Account Access": (2, 24),
    "Product Defect": (12, 72),
    "General Inquiry": (1, 12),
}


# One-sentence rationale for the ranges.
RESOLUTION_RANGE_RATIONALE = {
    "Billing": (
        "Billing issues normally require verification of payment "
        "and transaction information, so resolution is typically "
        "within one to three days."
    ),
    "Technical Issue": (
        "Technical issues can require investigation, troubleshooting, "
        "or coordination with a technical team, so they may take up "
        "to two days."
    ),
    "Account Access": (
        "Account-access problems are often resolved through identity "
        "verification and account checks, so they generally take "
        "less than one day."
    ),
    "Product Defect": (
        "Product defects may require investigation and coordination "
        "with the responsible team, so they can take several days."
    ),
    "General Inquiry": (
        "General inquiries are usually straightforward and can normally "
        "be resolved within the same business day."
    ),
}


# ---------------------------------------------------------
# 5. Generate categories
# ---------------------------------------------------------

ticket_categories = []

# Guarantee at least 3 records for every required category.
for category in CATEGORIES:
    ticket_categories.extend([category] * 3)


# Add remaining categories using the documented weights.
remaining_categories = TOTAL_TICKETS - len(ticket_categories)

weighted_categories = list(CATEGORY_WEIGHTS.keys())
weighted_category_values = list(CATEGORY_WEIGHTS.values())

for _ in range(remaining_categories):
    category = random.choices(
        weighted_categories,
        weights=weighted_category_values,
        k=1,
    )[0]

    ticket_categories.append(category)


# Shuffle so guaranteed records are not grouped together.
random.shuffle(ticket_categories)


# ---------------------------------------------------------
# 6. Generate statuses
# ---------------------------------------------------------

ticket_statuses = []

# Guarantee every required status appears at least once.
ticket_statuses.extend(STATUSES)

# Add remaining statuses using the documented weights.
remaining_statuses = TOTAL_TICKETS - len(ticket_statuses)

weighted_statuses = list(STATUS_WEIGHTS.keys())
weighted_status_values = list(STATUS_WEIGHTS.values())

for _ in range(remaining_statuses):
    status = random.choices(
        weighted_statuses,
        weights=weighted_status_values,
        k=1,
    )[0]

    ticket_statuses.append(status)


# Shuffle statuses.
random.shuffle(ticket_statuses)


# ---------------------------------------------------------
# 7. Create exactly 20% escalated tickets
# ---------------------------------------------------------

escalated_count = int(
    TOTAL_TICKETS * ESCALATION_RATE
)

escalated_ticket_numbers = set(
    random.sample(
        range(TOTAL_TICKETS),
        escalated_count,
    )
)


# ---------------------------------------------------------
# 8. Build final dataset
# ---------------------------------------------------------

SUPPORT_TICKETS = []

for i in range(TOTAL_TICKETS):

    category = ticket_categories[i]
    status = ticket_statuses[i]

    min_hours, max_hours = RESOLUTION_RANGES[category]

    resolution_time = round(
        random.uniform(
            min_hours,
            max_hours,
        ),
        1,
    )

    days_since_created = random.randint(
        0,
        30,
    )

    escalated = i in escalated_ticket_numbers

    # Keep status and escalation flag consistent.
    if escalated:
        status = "Escalated"

    elif status == "Escalated":
        status = "In Progress"

    ticket = {
        "record_id": f"TKT-{i + 1:04d}",
        "category": category,
        "status": status,
        "resolution_time_hours": resolution_time,
        "days_since_created": days_since_created,
        "escalated": escalated,
    }

    SUPPORT_TICKETS.append(ticket)


# ---------------------------------------------------------
# 9. Dataset validation
# ---------------------------------------------------------

def validate_dataset():

    assert len(SUPPORT_TICKETS) >= 40

    category_counts = Counter(
        ticket["category"]
        for ticket in SUPPORT_TICKETS
    )

    status_counts = Counter(
        ticket["status"]
        for ticket in SUPPORT_TICKETS
    )

    # Required categories must appear at least 3 times.
    for category in CATEGORIES:
        assert category_counts[category] >= 3

    # Required statuses must appear at least once.
    for status in STATUSES:
        assert status_counts[status] >= 1

    # Validate every ticket.
    for ticket in SUPPORT_TICKETS:

        assert isinstance(
            ticket["days_since_created"],
            int,
        )

        assert 0 <= ticket["days_since_created"] <= 30

        assert isinstance(
            ticket["escalated"],
            bool,
        )

        assert (
            ticket["record_id"].startswith("TKT-")
        )

    escalated_count = sum(
        ticket["escalated"]
        for ticket in SUPPORT_TICKETS
    )

    escalation_percentage = (
        escalated_count / len(SUPPORT_TICKETS)
    ) * 100

    # Required escalation range: 10%–30%.
    assert 10 <= escalation_percentage <= 30

    print("\nAll dataset validation checks passed.")


# ---------------------------------------------------------
# 10. Print dataset summary
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("OLA SUPPORT TICKET DATASET")
    print("=" * 60)

    print(f"\nSeed: {SEED}")
    print(f"Total tickets: {len(SUPPORT_TICKETS)}")

    print("\nCategory Weights")
    print("-" * 30)

    for category, weight in CATEGORY_WEIGHTS.items():
        print(f"{category:<20} {weight:.0%}")

    print("\nStatus Weights")
    print("-" * 30)

    for status, weight in STATUS_WEIGHTS.items():
        print(f"{status:<20} {weight:.0%}")

    print("\nCategory Counts")
    print("-" * 30)

    category_counts = Counter(
        ticket["category"]
        for ticket in SUPPORT_TICKETS
    )

    for category, count in sorted(
        category_counts.items()
    ):
        print(f"{category:<20} {count}")

    print("\nStatus Counts")
    print("-" * 30)

    status_counts = Counter(
        ticket["status"]
        for ticket in SUPPORT_TICKETS
    )

    for status, count in sorted(
        status_counts.items()
    ):
        print(f"{status:<20} {count}")

    escalated_count = sum(
        ticket["escalated"]
        for ticket in SUPPORT_TICKETS
    )

    escalated_percentage = (
        escalated_count / len(SUPPORT_TICKETS)
    ) * 100

    print("\nEscalation Summary")
    print("-" * 30)
    print(f"Escalated tickets: {escalated_count}")
    print(
        f"Escalated percentage: "
        f"{escalated_percentage:.1f}%"
    )

    print("\nResolution-Time Rationale")
    print("-" * 30)

    for category, rationale in (
        RESOLUTION_RANGE_RATIONALE.items()
    ):
        print(f"\n{category}:")
        print(f"  {rationale}")

    print("\nFirst 5 tickets")
    print("-" * 30)

    for ticket in SUPPORT_TICKETS[:5]:
        print(ticket)

    validate_dataset()