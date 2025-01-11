from google.api import annotations_pb2 as _annotations_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GetLabelsRequest(_message.Message):
    __slots__ = ("title", "body", "labels")
    TITLE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    title: str
    body: str
    labels: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, title: _Optional[str] = ..., body: _Optional[str] = ..., labels: _Optional[_Iterable[str]] = ...) -> None: ...

class GetLabelsResponse(_message.Message):
    __slots__ = ("labels",)
    LABELS_FIELD_NUMBER: _ClassVar[int]
    labels: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, labels: _Optional[_Iterable[str]] = ...) -> None: ...
