import pandas as pd
from datetime import datetime, timedelta
import random


def generate_transactions(start_date='2023-01-01', num_days=90, initial_balance=1837.23):
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    balance = initial_balance
    transactions = []

    for day in range(num_days):
        # Генерируем 0-5 транзакций в день
        num_transactions = random.randint(0, 15)

        for _ in range(num_transactions):
            # Определяем тип транзакции
            is_withdrawal = random.random() > 0.3  # 70% снятий, 30% пополнений

            if is_withdrawal:
                # Суммы снятия обычно небольшие
                withdrawal = round(random.uniform(5, 200), 2)
                deposit = 0
                balance -= withdrawal
            else:
                # Пополнения обычно больше
                withdrawal = 0
                deposit = round(random.uniform(50, 1500), 2)
                balance += deposit

            # Форматируем дату
            date_str = current_date.strftime('%d/%m/%Y')

            transactions.append({
                'date': date_str,
                'withdrawal': withdrawal,
                'deposit': deposit,
                'balance': round(balance, 2)
            })

        current_date += timedelta(days=1)

    return pd.DataFrame(transactions)


# Генерируем данные
df = generate_transactions(num_days=360)
print(df.head(20))
df.to_csv('tests/generated_transactions.csv', index=False)