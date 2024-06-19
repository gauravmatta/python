import pika
from pika.exchange_type import ExchangeType


def message_processor(ch, method, properties, body):
    if method.delivery_tag % 5 == 0:
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=True)
    if method.delivery_tag % 7 == 0 and not method.delivery_tag % 5 == 0:
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True, multiple=True)
    print(f"received new message: {body} with delivery Tag {method.delivery_tag}")


connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()

channel.exchange_declare(exchange='accept_reject_exchange', exchange_type=ExchangeType.fanout)

channel.queue_declare(queue='accept_reject_queue')
channel.queue_bind('accept_reject_queue', 'accept_reject_exchange', 'test')
channel.basic_consume(queue='accept_reject_queue', on_message_callback=message_processor)

print("Started Consuming")
channel.start_consuming()
