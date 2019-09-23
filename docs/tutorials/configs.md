# Using Configs

### Basics

### Best Practice with Configs

1. Treat the configs you write as "code": avoid copying them or duplicating them, but use "_BASE_"
	 instead to share common parts between configs.

2. Keep the configs you write simple: don't include keys that do not affect the job.

3. Keep a version number in your configs (or the base config), e.g., `VERSION: 1`. This way they can be automatically
	 upgraded when backward incompatible changes happen to the config definition.

4. Save a full config together with the model, and use it to run inference.
   This is more robust to changes that may happen to the config definition.
