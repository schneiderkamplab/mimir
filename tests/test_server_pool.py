import json
import pytest
from mimir.serving.server import ServerSpec,command


def test_server_compatibility_includes_template_contents(tmp_path):
    (tmp_path/'export_receipt.json').write_text('{}')
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'mimir2','mos_sparse':True}))
    template=tmp_path/'chat.jinja';template.write_text('first')
    spec=ServerSpec(str(tmp_path),'0',template=str(template))
    before=spec.key();template.write_text('second');assert spec.key()!=before
    with pytest.raises(ValueError,match='eager'):command(spec)
    assert '--enforce-eager' in command(ServerSpec(str(tmp_path),'0',eager=True))
