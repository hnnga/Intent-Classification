from flow.workflow import app


if __name__ == "__main__":
    user_question = input("User question: ")
    result = app.invoke({"question": user_question})

    print("\n === Result === ")
    print("Intent:", result.get("intent"))
    print("Action:", result.get("action"))
    print("Found in DB:", result.get("found_in_db"))

