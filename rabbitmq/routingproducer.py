import pika
from pika.exchange_type import ExchangeType

connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()
channel.exchange_declare(exchange='routing', exchange_type=ExchangeType.direct)
message = "This message is to be routed to "
channel.basic_publish(exchange='routing', routing_key='analyticsonly', body=message+"analytics")
channel.basic_publish(exchange='routing', routing_key='paymentsonly', body=message+"payments")
channel.basic_publish(exchange='routing', routing_key='both', body=message+"payments and analytics")
print(f"sent message: {message}")
connection.close()
