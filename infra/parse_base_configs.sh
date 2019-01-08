#!/bin/bash -e
# File: parse_base_configs.sh

# Print the path of the config file and its base config

CFG="$1"

if [[ -z "$CFG" ]]; then
	echo "Usage: $0 /path/to/config/file"
	exit 1
fi


function list-base() {
	# Args: path to config file
	local cfg="$1"
	local dir
	dir=$(dirname "$cfg")
	local base
	base=$(cat "$cfg" | grep "^_BASE_" | grep -o '".*"')
	if [[ -n "$base" ]]; then
		local path="${base//\"}"
		if ! [[ "$path" == /* ]]; then
			path="$dir/$path"
		fi
		if ! [[ -f "$path" ]]; then
			echo "File $path does not exist!" >&2
			exit 1
		fi
		echo "$path"
		list-base "$path"
	fi
}

echo "$CFG"
list-base "$CFG"
