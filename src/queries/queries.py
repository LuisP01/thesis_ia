OBTAIN_USERS= """
    LIMIT %s
FROM public.users u
WHERE EXISTS (
    SELECT 1 
    FROM public.bill b 
    WHERE b.client_id = u.id
)
AND NOT EXISTS (
    SELECT 1 
    FROM public.forecast f 
    WHERE f.user_id = u.id
)
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