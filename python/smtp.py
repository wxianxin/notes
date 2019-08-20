# Author: Steven Wang   Date: 20160724
# python 3.5.2

import smtplib

smtp_server = smtplib.SMTP("smtp.gmail.com:587")
from_addr = "stevenxsystem@gmail.com"
password = "zidongmima"
smtp_server.ehlo()
#ehlo: ehello, for eSTMP, extended SMTP
msg = "\r\n".join([
  "From: steven",
  "To: you",
  "Subject: This is an automated email",
  "",
  "This is an automated email."
  ])

smtp_server.starttls()
#Transport layer security
smtp_server.login(from_addr, password)
smtp_server.sendmail("sender",from_addr, msg)
smtp_server.close
