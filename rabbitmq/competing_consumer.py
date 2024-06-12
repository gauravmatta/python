import pika
import time
import random


def on_message_received(ch, method, properties, body):
    print(f" received new message: {body}")


def on_message_processed(ch, method, properties, body):
    processing_time = random.randint(1, 6)
    print(f" received : {body}, will take {processing_time} to process")
    time.sleep(processing_time)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print("Finished processing the message")


connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()
channel.queue_declare(queue='letterbox')
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='letterbox', auto_ack=True, on_message_callback=on_message_received)
channel.basic_consume(queue='letterbox', on_message_callback=on_message_processed)
print("Started Consuming")

channel.start_consuming()
