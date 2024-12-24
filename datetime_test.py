def my_function(my_int: int | None = None):
    """
    A function with an integer parameter that can be None.

    Args:
        my_int: An integer value or None. Defaults to None.
    """
    if my_int is None:
        print("my_int is None")
    else:
        print(f"my_int is: {my_int}")

my_function()
my_function(5)