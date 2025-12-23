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

