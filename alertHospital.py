from twilio.rest import Client


# Account Sid and Auth Token from twilio.com/console

def callHosp():

    account_sid = 'ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    auth_token = 'your_auth_token'
    client = Client(account_sid, auth_token)

    call = client.calls.create(
                url='http://demo.twilio.com/docs/voice.xml',
                to='+14155551212',
                from_='+15017122661'
            )

    #print(call.sid)