from app import app

# This file serves as the entry point for Gunicorn
# No need to run the app here as Gunicorn will handle that

if __name__ == '__main__':
    # This block is only for local development
    # and won't be used in production with Gunicorn
    import os
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)