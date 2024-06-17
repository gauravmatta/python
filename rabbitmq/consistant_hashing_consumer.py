import pika
from pika.exchange_type import ExchangeType


def on_message_letterbox_received(ch, method, properties, body):
    print(f" received new message in letterbox 1: {body}")


def on_message_letterbox2_received(ch, method, properties, body):
    print(f" received new message in letterbox 2: {body}")


def on_message_letterbox3_received(ch, method, properties, body):
    print(f" received new message in letterbox 2: {body}")


def on_message_letterbox4_received(ch, method, properties, body):
    print(f" received new message in letterbox 2: {body}")


connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()
channel.exchange_declare('hash_exchange', 'x-consistent-hash')
channel.queue_declare(queue='letterbox1')
channel.queue_bind('letterbox1', 'hash_exchange', routing_key='1')
channel.basic_consume(queue='letterbox1', auto_ack=True, on_message_callback=on_message_letterbox_received)
channel.queue_declare(queue='letterbox2')
channel.queue_bind('letterbox2', 'hash_exchange', routing_key='2')
channel.basic_consume(queue='letterbox2', auto_ack=True, on_message_callback=on_message_letterbox2_received)
channel.queue_declare(queue='letterbox3')
channel.queue_bind('letterbox3', 'hash_exchange', routing_key='3')
channel.basic_consume(queue='letterbox3', auto_ack=True, on_message_callback=on_message_letterbox3_received)
channel.queue_declare(queue='letterbox4')
channel.queue_bind('letterbox4', 'hash_exchange', routing_key='4')
channel.basic_consume(queue='letterbox4', auto_ack=True, on_message_callback=on_message_letterbox4_received)

print("Started Consuming")
channel.start_consuming()
