<!-- CONTENT_HASH: c7936fc795bd10426b0ca0efedfd2dd9f5d7db342e66f11d029ab4817ab1a7c0 -->

# Fence Parsing Fixture

This file exists to ensure SECTION_INDEX generation ignores headings inside fenced code blocks.

## Real Section

This should be indexed as a section when fixtures are explicitly included.

```python
# Fake Heading In Fence
print("do not index headings inside fences")
```

## Another Real Section

This should also be indexed as a section when fixtures are explicitly included.

~~~text
### Also Fake Inside Tilde Fence
~~~
