import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 💌 Configura tus credenciales aquí
EMAIL_USER = "sirem66@gmail.com"
EMAIL_PASS = "nygc cwba nkiu gmra"
EMAIL_TO = "sirem66@gmail.com"  # puedes poner el mismo o varios separados por coma

def enviar_alerta_email(asunto: str, mensaje: str):
    """Envía un correo con la alerta de trading."""
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_TO
        msg["Subject"] = asunto
        msg.attach(MIMEText(mensaje, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)

        print(f"📧 Alerta enviada: {asunto}")

    except Exception as e:
        print(f"⚠️ Error al enviar correo: {e}")
