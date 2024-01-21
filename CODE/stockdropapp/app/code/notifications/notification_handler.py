import plyer

def send_notification(message):
    plyer.notification.notify(
        title='Stock Drop App',
        message=message,
        timeout=10  # Notification timeout in seconds
    )
