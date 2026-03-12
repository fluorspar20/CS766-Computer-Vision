def sign_academic_honesty_policy(name: str, netid: str):
    """
    Prints an academic honesty statement or an error if not properly signed.
    """

    if name == "full_name" or netid == "netid":
        statement_str = "ERROR: Academic Honesty Policy agreement was not signed."
    else:
        statement_str = (
            f"I, {name} ({netid}), \n"
            "certify that I have read and agree to the Code of Academic Integrity."
        )

    header = "\n\n***********************\n"
    footer = "\n***********************\n\n"

    print(f"{header}{statement_str}{footer}")


if __name__ == "__main__":
    # sign_academic_honesty_policy("full_name", "netid")
    pass
