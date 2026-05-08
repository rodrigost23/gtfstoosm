import datetime
import logging
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OSMElement(BaseModel):
    id: int
    version: int = 1
    changeset: int | None = None
    timestamp: datetime.datetime | str | None = None
    user: str | None = None
    uid: int | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    original_tags: dict[str, str] = Field(default_factory=dict)
    force_conflict: bool = False

    def is_functionally_changed(self) -> bool:
        """Check if the element has functional changes from its original state."""
        return self.tags != self.original_tags

    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the element."""
        if key in self.tags:
            if self.tags[key] != value:
                raise ValueError(f"Key {key} already exists with different value: {self.tags[key]}")
            return
        if value and value != "":
            self.tags[key] = value

    def modify_tag(self, key: str, value: str) -> None:
        """Modify a tag in the element."""
        if key not in self.tags:
            raise ValueError(f"Key {key} does not yet exist")
        if value and value != "":
            self.tags[key] = value

    def tags_to_xml(self) -> str:
        """Create an XML tag element."""
        return "\n".join(
            [f'<tag k="{key}" v="{value}"></tag>' for key, value in self.tags.items()]
        )

    def get_base_attributes(self) -> str:
        """Get standard OSM XML attributes."""
        version = "1" if self.force_conflict else self.version
        attrs = [f'id="{self.id}"', f'version="{version}"']
        if self.changeset:
            attrs.append(f'changeset="{self.changeset}"')
        if self.timestamp:
            ts = self.timestamp
            if isinstance(ts, datetime.datetime):
                ts = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            attrs.append(f'timestamp="{ts}"')
        if self.user:
            attrs.append(f'user="{self.user}"')
        if self.uid:
            attrs.append(f'uid="{self.uid}"')
        return " ".join(attrs)


class OSMNode(OSMElement):
    lat: float
    lon: float
    visible: bool = True

    def to_xml(self) -> str:
        osm_text = f"<node {self.get_base_attributes()} lat=\"{self.lat}\" lon=\"{self.lon}\">"
        if self.tags:
            osm_text += "\n" + self.tags_to_xml() + "\n</node>"
        else:
            osm_text += " />"
        return osm_text


class OSMWay(OSMElement):
    nodes: list[int] = Field(default_factory=list)
    visible: bool = True

    def add_node(self, node_id: int) -> None:
        """Add a node to the way."""
        self.nodes.append(node_id)

    def to_xml(self) -> str:
        osm_text = f"<way {self.get_base_attributes()} visible=\"{str(self.visible).lower()}\">\n"
        osm_text += "\n".join([f"<nd ref='{node_id}'></nd>" for node_id in self.nodes])
        if self.tags:
            osm_text += "\n" + self.tags_to_xml()
        osm_text += "\n</way>"


class RelationMember(BaseModel):
    type: Literal["node", "way", "relation"]
    ref: int
    role: str

    def to_xml(self) -> str:
        """Create an XML member element."""
        return (
            f'<member type="{self.type}" ref="{self.ref}" role="{self.role}"></member>'
        )


class OSMRelation(OSMElement):
    members: list[RelationMember] = Field(default_factory=list)
    original_members: list[RelationMember] = Field(default_factory=list)
    visible: bool = True

    def is_functionally_changed(self) -> bool:
        """Check if the relation has functional changes from its original state."""
        # Check tags first
        if self.tags != self.original_tags:
            return True
        # Check members (order matters for functional equality)
        if len(self.members) != len(self.original_members):
            return True
        for i, member in enumerate(self.members):
            orig = self.original_members[i]
            if member.type != orig.type or member.ref != orig.ref or member.role != orig.role:
                return True
        return False

    def add_member(
        self, osm_type: Literal["node", "way", "relation"], ref: int, role: str = ""
    ) -> None:
        """Add a member to the relation."""
        # Validate type
        if osm_type not in ["node", "way", "relation"]:
            raise ValueError(f"Invalid member type: {osm_type}")

        self.members.append(RelationMember(type=osm_type, ref=ref, role=role))

    def to_xml(self) -> str:
        """Create an XML relation element."""
        osm_text = f"<relation {self.get_base_attributes()} visible=\"{str(self.visible).lower()}\">\n"

        osm_text += "\n".join([member.to_xml() for member in self.members])
        if self.tags:
            osm_text += "\n" + self.tags_to_xml()

        return osm_text + "\n</relation>"
