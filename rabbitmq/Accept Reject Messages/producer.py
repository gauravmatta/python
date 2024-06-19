import pika
from pika.exchange_type import ExchangeType

connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()

channel.exchange_declare(exchange='accept_reject_exchange', exchange_type=ExchangeType.fanout)

i = 1
while True:
    message = "Lets send Message Number " + str(i)
    i = i + 1
    channel.basic_publish(exchange='accept_reject_exchange', routing_key='test', body=message)
    print(f"sent message: {message}")
    input('Press any Key to continue..')
