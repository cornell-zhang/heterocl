import heterocl as hcl

def test_issue_410():

    A = hcl.Struct ({'foo': hcl.UInt(16) })
    B = hcl.Struct ({'foo': 'uint16' })

    assert A['foo'] == B['foo']
