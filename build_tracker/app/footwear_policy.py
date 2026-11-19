"""Explicit footwear-state contract shared by isolated training and both policies."""
FOOTWEAR_STATE_VERSION = 'footwear-timer-slots-v1'


def validate_footwear_state(row, *, required_version=None):
    version=row.get('footwear_state_version')
    if version is None and required_version is None:
        return False
    if version!=FOOTWEAR_STATE_VERSION or required_version not in (None,version):
        raise ValueError('Footwear state is missing or differs from the artifact schema')
    for member in (row,*(row.get('others') or [])):
        if member.get('footwear_state_version')!=version or any(type(member.get(key)) is not bool for key in
                ('magical_footwear_selected','footwear_granted')):
            raise ValueError('Explicit boolean footwear state is required for shopper and board')
        if member['footwear_granted'] and not member['magical_footwear_selected']:
            raise ValueError('A footwear grant requires the selected rune')
    return True


def footwear_arguments(row, dragon, *, required_version=None):
    if not validate_footwear_state(row,required_version=required_version):
        return {}
    if getattr(dragon,'footwear_state_version',None)!=FOOTWEAR_STATE_VERSION:
        raise ValueError('Footwear-aware rows require the explicitly activated economics runtime')
    return {'magical_footwear_selected':row['magical_footwear_selected'],
            'footwear_granted':row['footwear_granted']}
