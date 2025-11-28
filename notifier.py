# notifier.py
import smtplib
import socket
import time
from email.message import EmailMessage
import config

class Notificador:
    """
    Classe responsável por gerenciar o envio de alertas externos.
    """

    @staticmethod
    def enviar_email(assunto, corpo_mensagem):
        # Prepara a mensagem
        msg = EmailMessage()
        msg['Subject'] = assunto
        msg['From'] = config.EMAIL_REMETENTE
        msg['To'] = ", ".join(config.EMAIL_DESTINATARIOS)
        msg.set_content(corpo_mensagem)

        for tentativa in range(1, config.EMAIL_MAX_TENTATIVAS + 1):
            try:
                print(f"\n[Email] Conectando ao servidor... (Tentativa {tentativa}/{config.EMAIL_MAX_TENTATIVAS})")
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(config.EMAIL_REMETENTE, config.EMAIL_SENHA_APP)
                    smtp.send_message(msg)
                print("[Email] Sucesso! Alerta enviado.")
                return  # SUCESSO

            except (socket.gaierror, smtplib.SMTPException, TimeoutError) as e:
                print(f"[Email] Erro de Rede (Tentativa {tentativa}): {e}")
                if tentativa < config.EMAIL_MAX_TENTATIVAS:
                    print(f"[Email] Aguardando {config.EMAIL_ESPERA_SEGUNDOS}s...")
                    time.sleep(config.EMAIL_ESPERA_SEGUNDOS)
            
            except Exception as e:
                print(f"[Email] ERRO CRÍTICO (Auth ou Outro): {e}")
                return

        print("[Email] Falha total no envio.")