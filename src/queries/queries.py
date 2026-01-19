OBTAIN_USERS= """
    select username from users
    LIMIT %s
"""