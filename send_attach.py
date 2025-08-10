import base64
import mimetypes
import os
from email.message import EmailMessage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

import google.auth
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# 1. Define the scopes (permissions) we need.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send"
]

def gmail_create_draft_with_attachment(creds):
    """
    Create a draft email with attachment, then return the draft object.
    """
    try:
        # Build the Gmail API client
        service = build("gmail", "v1", credentials=creds)
        mime_message = EmailMessage()

        # Email headers
        mime_message["To"] = "wilkpaulc@gmail.com"
        mime_message["From"] = "wilkpaulc@gmail.com"
        mime_message["Subject"] = "Sample with attachment"

        # Email text
        mime_message.set_content(
            "Hello,\n\nThis is an automated mail with attachment.\nPlease do not reply."
        )

        # Identify the path and attach a file
        attachment_filename = "stock_project/plots/close_2025-01-29.png"
        type_subtype, _ = mimetypes.guess_type(attachment_filename)
        maintype, subtype = type_subtype.split("/")

        with open(attachment_filename, "rb") as fp:
            attachment_data = fp.read()
        mime_message.add_attachment(
            attachment_data, 
            maintype, 
            subtype, 
            filename="sp_image.png"
        )

        # Encode in base64
        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
        create_draft_request_body = {"message": {"raw": encoded_message}}

        # Create the draft in Gmail
        draft = (
            service.users()
                   .drafts()
                   .create(userId="me", body=create_draft_request_body)
                   .execute()
        )
        print(f"Draft created. ID: {draft['id']}")
        return draft

    except HttpError as error:
        print(f"An error occurred while creating the draft: {error}")
        return None

def gmail_send_draft(draft_id, creds):
    """
    Send an existing draft by its draft_id.
    """
    try:
        service = build("gmail", "v1", credentials=creds)
        sent_message = (
            service.users()
                   .drafts()
                   .send(userId="me", body={"id": draft_id})
                   .execute()
        )
        print(f"Draft with ID {draft_id} has been sent successfully.")
        return sent_message

    except HttpError as error:
        print(f"An error occurred while sending the draft: {error}")
        return None

def main():
    # 2. Handle credentials (token.json is automatically created after first authorization).
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    # 3. Create a draft with attachment
    draft = gmail_create_draft_with_attachment(creds)
    if draft is None:
        print("Failed to create draft.")
        return

    # 4. Send the draft
    draft_id = draft["id"]
    sent_msg = gmail_send_draft(draft_id, creds)
    if sent_msg:
        print("Email successfully sent.")
    else:
        print("Failed to send the email.")

if __name__ == "__main__":
    main()
