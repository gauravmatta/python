import pika
from pika.exchange_type import ExchangeType

connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
channel = connection.channel()
# Enable Publisher Confirms
channel.confirm_delivery()
# Enables Transactions
channel.tx_select()

channel.exchange_declare(exchange='pubsub', exchange_type=ExchangeType.fanout)
# Creates a durable queue
channel.queue_declare('Test', durable=True)
message = "Hello I want to broadcast this message"
channel.basic_publish(exchange='pubsub', routing_key='', properties=pika.BasicProperties(
    headers={'name': 'gaurav'},
    delivery_mode=1,
    expiration=13434343,
    content_type="application/json"
),
                      body=message,
                      # Set the publishing to be mandatory i.e. receive a notification of failure.
                      mandatory=True
                      )
# Commit the transaction
channel.tx_commit()
# Roll Back Transaction
channel.tx_rollback()

connection.close()