OBTAIN_USERS= """
    select id, username, water_data, electricity_data from users
    where id = 2
"""

INSERT_FORECAST="""
    insert into
    FORECAST(type, period_date, amount, payment, suggestion, user_id, is_read, superior_interval, inferior_interval, predict_percentage)
    VALUES(
        %s,
        %s,
        %s,
        %s,
        null,
        %s,
        'false',
        %s,
        %s,
        %s
    )
"""