from tool import slugify
def test_slugify():
    assert slugify("Hello World") == "hello-world"
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  Space   Test  ") == "space-test"
    assert slugify("Special @#$% Characters" ) == "special-characters"
    assert slugify("Numbers 123 are allowed") == "numbers-123-are-allowed"
    assert slugify("Leading and trailing - hyphens -") == "leading-and-trailing-hyphens"
    assert slugify("Multiple   Spaces") == "multiple-spaces"
    print("All tests passed!")

if __name__ == "__main__":
    test_slugify()
