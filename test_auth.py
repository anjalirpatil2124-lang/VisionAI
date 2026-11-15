from getpass import getpass

from database import create_user, authenticate_user


print("\n========== VisionAI Authentication Test ==========")

print("\n1. USER REGISTRATION")

username = input("Enter a test username: ").strip()
password = getpass("Enter a test password (minimum 8 characters): ")

success, message = create_user(username, password)

print(message)

if success:

    print("\n2. USER LOGIN TEST")

    login_password = getpass("Enter the same password to login: ")

    user = authenticate_user(username, login_password)

    if user:
        print("\nLogin successful!")
        print("User ID:", user["id"])
        print("Username:", user["username"])

    else:
        print("\nLogin failed!")

else:
    print("\nRegistration test could not continue.")

print("\n========== Test Completed ==========")