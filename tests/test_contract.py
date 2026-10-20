import json
import pytest
from mimir.checkpoints.contract import publish,validate,resolve


def test_partial_corrupt_and_weights_only_rejected(tmp_path):
    stage=tmp_path/'partial';stage.mkdir();(stage/'state').write_text('saved')
    with pytest.raises(FileNotFoundError):validate(stage)
    out=tmp_path/'step_1';publish(stage,out,metadata={'mode':'weights','tag':'step_1'})
    validate(out)
    with pytest.raises(ValueError,match='Weights-only'):resolve(out,'step_1')
    (out/'state').write_text('wrong')
    with pytest.raises(ValueError,match='Corrupt'):validate(out)


def test_publication_does_not_replace_existing_checkpoint(tmp_path):
    out=tmp_path/'step_1';out.mkdir();(out/'old').write_text('safe')
    stage=tmp_path/'partial';stage.mkdir();(stage/'new').write_text('new')
    with pytest.raises(FileExistsError):publish(stage,out,metadata={})
    assert (out/'old').read_text()=='safe'
