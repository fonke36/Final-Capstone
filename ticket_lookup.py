"""
Part 2.6 - Support Ticket Lookup Tool

Looks up an Ola support ticket and calculates an escalation score.

Escalation score formula:
    escalation_score =
        0.60 * escalated_flag
        + 0.40 * recency_score

where:
    recency_score = 1 - (days_since_created / 30)

The escalation flag has more weight because an explicitly escalated
ticket is a stronger signal than recency alone.
"""

from dataset import SUPPORT_TICKETS


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

MAX_DAYS = 30

ESCALATED_WEIGHT = 0.60
RECENCY_WEIGHT = 0.40


# ---------------------------------------------------------
# 2. Calculate escalation score
# ---------------------------------------------------------

def calculate_escalation_score(ticket: dict) -> float:
    """
    Calculate an escalation score between 0 and 1.
    """

    escalated_flag = 1 if ticket["escalated"] else 0

    recency_score = 1 - (
        ticket["days_since_created"] / MAX_DAYS
    )

    score = (
        ESCALATED_WEIGHT * escalated_flag
        + RECENCY_WEIGHT * recency_score
    )

    return round(score, 4)


# ---------------------------------------------------------
# 3. Analyze score distribution
# ---------------------------------------------------------

def analyze_score_distribution():
    """
    Calculate score statistics for escalated and
    non-escalated tickets.
    """

    escalated_scores = []
    non_escalated_scores = []

    for ticket in SUPPORT_TICKETS:
        score = calculate_escalation_score(ticket)

        if ticket["escalated"]:
            escalated_scores.append(score)
        else:
            non_escalated_scores.append(score)

    max_non_escalated = max(non_escalated_scores)
    min_escalated = min(escalated_scores)

    # Threshold is placed between the two observed groups.
    threshold = round(
        (max_non_escalated + min_escalated) / 2,
        4
    )

    print("\nEscalation Score Distribution")
    print("-" * 40)

    print(
        f"Non-escalated tickets: "
        f"min={min(non_escalated_scores):.4f}, "
        f"max={max_non_escalated:.4f}, "
        f"avg={sum(non_escalated_scores) / len(non_escalated_scores):.4f}"
    )

    print(
        f"Escalated tickets: "
        f"min={min_escalated:.4f}, "
        f"max={max(escalated_scores):.4f}, "
        f"avg={sum(escalated_scores) / len(escalated_scores):.4f}"
    )

    print(f"\nMaximum non-escalated score: {max_non_escalated:.4f}")
    print(f"Minimum escalated score:     {min_escalated:.4f}")
    print(f"Recommended threshold:       {threshold:.4f}")

    print(
        "\nThreshold justification:"
        "\nThe threshold is placed midway between the highest "
        "non-escalated score and the lowest escalated score "
        "observed in the generated dataset."
    )

    return threshold


# ---------------------------------------------------------
# 4. Support ticket lookup function
# ---------------------------------------------------------

def check_support_ticket_status(record_id: str) -> dict:
    """
    Look up a support ticket by record_id.

    Returns:
        status
        resolution_time_hours
        escalation_score
    """

    for ticket in SUPPORT_TICKETS:

        if ticket["record_id"] == record_id:

            score = calculate_escalation_score(ticket)

            return {
                "record_id": ticket["record_id"],
                "status": ticket["status"],
                "resolution_time_hours": ticket[
                    "resolution_time_hours"
                ],
                "escalation_score": score,
            }

    return {
        "error": f"Ticket {record_id} was not found."
    }


# ---------------------------------------------------------
# 5. Test the tool
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("PART 2.6 - SUPPORT TICKET LOOKUP TOOL")
    print("=" * 60)

    print("\nScoring Formula:")
    print(
        "escalation_score = "
        "0.60 × escalated_flag + "
        "0.40 × recency_score"
    )

    print(
        "recency_score = "
        "1 - (days_since_created / 30)"
    )

    print(
        "\nDataset size:",
        len(SUPPORT_TICKETS)
    )

    # Analyze dataset distribution
    threshold = analyze_score_distribution()

    # Test a valid ticket
    print("\n" + "-" * 40)
    print("Valid Ticket Test")
    print("-" * 40)

    valid_result = check_support_ticket_status("TKT-0001")

    print(valid_result)

    # Test another ticket
    print("\nSecond Valid Ticket Test")

    second_result = check_support_ticket_status("TKT-0002")

    print(second_result)

    # Test invalid ticket
    print("\n" + "-" * 40)
    print("Invalid Ticket Test")
    print("-" * 40)

    invalid_result = check_support_ticket_status("TKT-9999")

    print(invalid_result)

    # Show threshold decision
    print("\n" + "-" * 40)
    print("Threshold Decision Test")
    print("-" * 40)

    for ticket in SUPPORT_TICKETS[:5]:

        score = calculate_escalation_score(ticket)

        decision = (
            "HIGH ESCALATION RISK"
            if score >= threshold
            else "NORMAL"
        )

        print(
            f"{ticket['record_id']} | "
            f"score={score:.4f} | "
            f"{decision}"
        )

    print("\nPart 2.6 standalone test completed.")