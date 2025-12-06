from enum import StrEnum


class UserRole(StrEnum):
    USER = 'user'
    ADMIN = 'admin'


class TransactionCategory(StrEnum):
    FOOD = "Food"
    MISC = "Misc"
    RENT = "Rent"
    SALARY = "Salary"
    SHOPPING = "Shopping"
    TRANSPORT = "Transport"