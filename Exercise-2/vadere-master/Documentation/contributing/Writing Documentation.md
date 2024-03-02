<style>
.alert-warning {
  color: rgb(162,82,0) !important;
  padding: 10px;
    background-color: rgb(255,147,38);
    border-radius: 5px;
    border-color: rgb(162,82,0);
}
</style>

# Writing Documentation
The documentation is separated in two parts: 
- documentation for the users. 
- and documentation for the developer.

If you want to document a new feature, please follow the instructions below.

## Documentation for the User
The documentation for the user is automatically generated from the JavaDoc comments in the code.
Therefore, any comment that uses JavaDoc syntax will be included in the documentation.

See 'VadereGui/resources/js' for custom web components that can be used inside teh JavaDoc comments.

The user usually interacts with the simulator via editing the AttributesXXX.java classes. 
Therefore, those classes are the first place to add user documentation.

You can add links to other classes by using the following syntax:
```java
{@link org.vadere.module.AClass}
```
To prevent circular dependencies, those link must be the full classpath.

## Documentation for the Developer
For writing documentation for the developer, we use the ``//` comments.
