import smtplib
import os
from dotenv import load_dotenv

load_dotenv()

smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = os.getenv("SMTP_EMAIL")
sender_password = os.getenv("SMTP_PASSWORD")

print(f"Testing SMTP login for: {sender_email} with password: {sender_password}")
try:
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)
    server.quit()
    print("SUCCESS: SMTP login successful!")
except Exception as e:
    print(f"FAILED: {e}")
