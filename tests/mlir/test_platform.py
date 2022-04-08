import heterocl as hcl
from heterocl.platforms import import_json_platform


def test_print_platform_hierarchy():
    import_json_platform("test_platform_spec/xilinx_u280.json")

if __name__ == "__main__":
    test_print_platform_hierarchy()