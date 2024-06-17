import pika
from pika.exchange_type import ExchangeType


def on_message_received(ch, method, properties, body):
    print(f" received new message: {body}")


connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()
channel.exchange_declare('headers_exchange', exchange_type=ExchangeType.headers)
channel.queue_declare(queue='letterbox')
bind_args = {
    'x-match': 'any',
    'name': 'gaurav',
    'age': '37'
}
channel.queue_bind('letterbox', 'headers_exchange', arguments=bind_args)
channel.basic_consume(queue='letterbox', auto_ack=True, on_message_callback=on_message_received)
print("Started Consuming")
channel.start_consuming()
