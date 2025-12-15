import pkg_resources

with open('requirements.txt', 'w') as f:
    for pkg in pkg_resources.working_set:
        f.write(f"{pkg.key}=={pkg.version}\n")

print("Requirements exported successfully!")