from typing import List

from prefect import flow, task
from prefect_email import EmailServerCredentials, email_send_message


@task
def create_prefect_email_block():
    credentials = EmailServerCredentials(
        username="sergiozoomcamp@gmail.com",
        password="aqmchfdpixqdjazp",  # must be an app password
    )
    credentials.save("EMAIL-BLOCK")


@flow
def example_email_send_message_flow(email_addresses: List[str]):
    email_server_credentials = EmailServerCredentials.load("EMAIL-BLOCK")
    for email_address in email_addresses:
        subject = email_send_message.with_options(name=f"email {email_address}").submit(
            email_server_credentials=email_server_credentials,
            subject="Example Flow Notification using Gmail",
            msg="This proves email_send_message works!",
            email_to=email_address,
        )