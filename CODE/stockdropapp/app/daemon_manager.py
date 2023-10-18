import threading
import time
from app.notification_handler import send_notification  # Implement this function

# Daemon function for a specific currency and time period
def daemon(currency, time_period, model):
    while True:
        # Daemon logic here
        send_notification(f'Daemon for {currency}, {time_period}, {model} running...')
        time.sleep(60)  # Example: Run every minute

# Function to start a daemon
def start_daemon(currency, time_period, model):
    daemon_thread = threading.Thread(target=daemon, args=(currency, time_period, model))
    daemon_thread.daemon = True
    daemon_thread.start()

# Inside daemon_manager.py
def stop_daemon(key, running_daemons):
    # Implement logic to stop the daemon based on the key
    # This function should stop the specified daemon
    # Get the thread associated with the key
    daemon_thread = running_daemons.get(key)
    if daemon_thread:
        # Terminate the thread
        daemon_thread._stop()
        # Remove the reference from the dictionary
        del running_daemons[key]

# # Example usage
# if __name__ == '__main__':
#     start_daemon('BTC', '1 day', 'Model1')
