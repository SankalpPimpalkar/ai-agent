from tool import password_generator
assert "Password: " in password_generator(length=10)
assert len(password_generator(length=10).split('\n')[0].split(': ')[1]) == 10
