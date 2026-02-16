from setuptools import setup

# 包名：安装后可通过 `import mbrl` 使用。
package_name = "mbrl"

setup(
    # Python 包的名字
    name=package_name,
    # 需要打包的 Python 模块目录
    packages=[package_name],
    # data_files 常用于 ROS/ament 生态中的包索引与元数据安装
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    # 安装后可直接调用的命令行入口
    # 例如：main -> mbrl.main:main
    entry_points={
        "console_scripts": [
            "main = " + package_name + ".main:main",
        ],
    },
)
