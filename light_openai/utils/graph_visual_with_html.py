# 导入pipmaster库，用于简化Python包的安装和管理
import pipmaster as pm
 
# 检查并安装pyvis库，如果尚未安装
if not pm.is_installed("pyvis"):
    pm.install("pyvis")
# 检查并安装networkx库，如果尚未安装
if not pm.is_installed("networkx"):
    pm.install("networkx")
 
# 导入networkx库，用于创建和操作复杂的网络图
import networkx as nx
# 从pyvis.network模块导入Network类，用于在浏览器中展示网络图
from pyvis.network import Network
# 导入random模块，用于生成随机数
import random
 
# 读取GraphML格式的图文件，创建一个NetworkX图对象G
G = nx.read_graphml("./dickens/graph_chunk_entity_relation.graphml")
 
# 创建一个Pyvis的Network对象，用于在浏览器中展示网络图
net = Network(height="100vh", notebook=True)
 
# 将NetworkX图对象G转换为Pyvis网络
net.from_nx(G)
 
# 遍历网络中的所有节点，为每个节点添加随机颜色和提示信息
for node in net.nodes:
    # 为节点分配一个随机颜色
    node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    # 如果节点有"description"属性，将其设置为鼠标悬停时显示的提示信息
    if "description" in node:
        node["title"] = node["description"]
 
# 遍历网络中的所有边，为每条边添加提示信息
for edge in net.edges:
    # 如果边有"description"属性，将其设置为鼠标悬停时显示的提示信息
    if "description" in edge:
        edge["title"] = edge["description"]
 
# 将网络图保存为HTML文件，并在浏览器中展示
net.show("../static/knowledge_graph.html")